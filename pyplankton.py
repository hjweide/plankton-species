import sys
import numpy as np
import sqlite3
from flask import Flask, Response
from flask import request, session, g, redirect, url_for, abort
from flask import send_from_directory
from flask import render_template, flash

from PIL import Image
import StringIO
import json
from os.path import join, dirname, isdir

# configuration
#DATABASE = '/tmp/pyplankton.db'
DATABASE = 'demo.db'
USERNAME = 'admin'
PASSWORD = 'default'

WIDTH, HEIGHT = 95, 95

app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/<path:filename>')
def image(filename):
    # when doing the overlay we want the full resolution image
    thumbnail = True
    try:
        thumbnail = request.args['thumbnail'] == 'True'
    except (KeyError, ValueError):
        pass

    try:
        im = Image.open(filename)
        if thumbnail:
            im.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)


# entry point when app is started
@app.route('/', methods=['GET'])
def home():
    cur = g.db.execute(
        'select species_name, count(image_id) '
        'from species, image '
        'where species_id=image_species_id '
        'group by species_id '
        'order by count(image_id) desc'
    )
    result = cur.fetchall()
    species_counts = []
    for (species_name, count_image_id) in result:
        #species_counts.append({
        #    'species_name': str(species_name),
        #    'image_count': count_image_id,
        #})
        species_counts.append([
            str(species_name),
            count_image_id
        ])

    return render_template('home.html',
                           species_counts=species_counts)


# when user chooses a new species for a set of images
@app.route('/label', methods=['POST'])
def post_labels():
    #print('POST to label_images')
    image_id_string = request.form['image_id']
    species_id_string = request.form['species_id']
    image_id_list = image_id_string.split(', ')

    values = []
    for image_id in image_id_list:
        values.append((
            species_id_string,
            image_id,
        ))
    cur = g.db.executemany(
        'update image set image_species_id=? where image_id=?;',
        values
    )
    g.db.commit()

    return json.dumps({'status': 'OK', 'rows_updated': cur.rowcount})


# called to overlay the native resolution of the selected image
@app.route('/overlay', methods=['POST'])
def post_overlay():
    #print('POST to overlay_image')
    species_id_string = request.form['image_id'],
    cur = g.db.execute(
        'select image_id, image_filepath, image_height, image_width,'
        ' species_name, user_username '
        'from image, species, user '
        'where image_id=? and '
        'image_species_id=species_id and '
        'image_user_id=user_id',
        species_id_string,
    )

    (image_id, image_filepath, image_height,
        image_width, species_name, user_username) = cur.fetchone()

    return json.dumps({
        'status': 'OK',
        'image_id': image_id,
        'image_filepath': image_filepath,
        'image_width': image_width,
        'image_height': image_height,
        'species_name': species_name,
        'username': user_username
    })


# when user chooses to work on labeling images manually
@app.route('/label')
def label_images():
    cur = g.db.execute(
        'select image_id, image_filepath, species_name from image, species'
        ' where image_species_id = species_id'
        ' order by random()'
        ' limit 1000'  # TODO: just for testing
    )
    result = cur.fetchall()

    images = []
    width, height = 95, 95
    for (image_id, image_filepath, species_name) in result:
        images.append({
            'image_id': image_id,
            'width': width,
            'height': height,
            'species': str(species_name),
            'src': image_filepath,
        })

    cur = g.db.execute(
        'select species_id, species_name '
        'from species '
        'order by species_name;'
    )
    result = cur.fetchall()

    labels = []
    for (species_id, species_name) in result:
        labels.append({
            'label_id': species_id,
            'label_name': str(species_name),
        })

    return render_template('label_images.html',
                           #images=map(json.dumps, images),
                           images=images,
                           labels=labels)


# when user chooses to review species
@app.route('/review')
def review_images():
    cur = g.db.execute(
        'select species_name from species order by species_id'
    )
    result = cur.fetchall()
    species = []
    for (species_name,) in result:
        species.append({
            'species_name': species_name,
        })

    return render_template('review_images.html',
                           species=species)


# prepares to query the database for images to be reviewed
@app.route('/prepare', methods=['POST'])
def prepare_review():
    print('prepare_review')
    limit_string = str(request.form['limit'])
    cur = g.db.execute(
        'select count(image_id) '
        'from image '
        'limit ?', (limit_string,)
    )

    result = cur.fetchone()[0]

    return json.dumps({'count': result})


@app.route('/update', methods=['POST'])
def post_revisions():
    image_id_string = request.form['image_id']
    species_name_string = request.form['species_name']

    cur = g.db.execute(
        'update image set image_species_id=('
        'select species_id from species where species_name=?) '
        'where image_id=?',
        (species_name_string, image_id_string)
    )

    g.db.commit()

    if cur.rowcount == 1:
        return json.dumps({'status': 'OK'})
    else:
        return json.dumps({'status': 'ERROR'})


@app.route('/review', methods=['POST'])
def review_annotations():
    # TODO: create minibatches and pass to model
    def get_class_scores(filenames, species):
        y_hat = np.random.lognormal(1., 5.,
                                    size=((len(filenames), len(species))))
        y_hat /= y_hat.sum(axis=1).reshape(-1, 1)
        return y_hat

    print('review_annotations')
    limit_string = str(request.form['limit'])
    cur = g.db.execute(
        'select image_id, image_filepath, image_height, image_width, '
        'species_name, user_username '
        'from image, species, user '
        'where image_species_id=species_id and '
        'image_user_id=user_id '
        'limit ?', (limit_string,)
    )

    result = cur.fetchall()

    cur = g.db.execute(
        'select species_name from species'
    )
    species = [s[0] for s in cur.fetchall()]
    image_filepaths = [str(x[1]) for x in result]

    class_scores = get_class_scores(image_filepaths, species)

    values = []
    for class_score, (image_id, image_filepath, image_height, image_width,
                      species_name, user_username) in zip(class_scores, result):
        scores_sorted, species_sorted = zip(*sorted(zip(class_score, species)))
        score_tuples = [(sc, sp) for sc, sp in zip(
            scores_sorted, species_sorted)][::-1]
        values.append({
            'image_id': image_id,
            'image_scores': score_tuples,
            'image_filepath': image_filepath,
            'image_height': image_height,
            'image_width': image_width,
            'species_name': species_name,
            'user_username': user_username,
        })

    return json.dumps(values)


def connect_db():
    return sqlite3.connect(app.config['DATABASE'])


@app.before_request
def before_request():
    g.db = connect_db()


@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()


def install_secret_key(app, filename='secret_key'):
    # adapted from: http://flask.pocoo.org/snippets/104/
    filename = join(app.instance_path, filename)
    try:
        app.config['SECRET_KEY'] = open(filename, 'rb').read()
    except IOError:
        print 'Error: No secret key. Create it with:'
        if not isdir(dirname(filename)):
            print('mkdir -p %s' % dirname(filename))
        print('head -c 24 /dev/urandom > %s' % filename)
        sys.exit(1)


if __name__ == '__main__':
    app.debug = True
    install_secret_key(app, filename='secret_key')
    app.run()
