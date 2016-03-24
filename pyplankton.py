import sys
import numpy as np
import sqlite3
import warnings
from flask import Flask, Response
from flask import request, session, g, redirect, url_for, abort
from flask import send_from_directory
from flask import render_template, flash

from PIL import Image
import StringIO
import json
from time import strftime
from os.path import join, dirname, isdir, isfile

# configuration is done in instance/default.py
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
            im.thumbnail(
                (app.config['WIDTH'], app.config['HEIGHT']), Image.ANTIALIAS)
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
        'select '
        '  species.species_name, species.species_confusable, '
        '  count(image.image_id) '
        'from species '
        'join image on '
        '   species.species_id=image.image_species_id '
        'group by species.species_id '
        'order by count(image.image_id) desc'
    )
    result = cur.fetchall()
    species_counts = []
    for (species_name, species_confusable, count_image_id) in result:
        species_counts.append({
            'species_name': str(species_name),
            'species_confusable': species_confusable,
            'count_image_id': count_image_id,
        })

    cur = g.db.execute(
        'select count(image_id) '
        'from image '
        'where image_species_id is not null'
    )

    annotated = cur.fetchone()[0]
    cur = g.db.execute(
        'select count(image_id) '
        'from image'
    )
    total = cur.fetchone()[0]

    return render_template('home.html',
                           annotated=annotated,
                           total=total,
                           species_counts=species_counts)


@app.route('/home/update', methods=['POST'])
def home_update():
    species_name_string = request.form['species_name']
    species_confusable_string = request.form['species_confusable']
    species_confusable = 1 if species_confusable_string == 'true' else 0
    g.db.execute(
        'update'
        '  species '
        'set '
        '  species_confusable=? '
        'where'
        '  species_name=?',
        (species_confusable, species_name_string)
    )

    g.db.commit()

    return json.dumps({'status': 'OK'})


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
        'select'
        '  image.image_id, image.image_filepath, '
        '  image.image_date_added, image.image_date_collected, '
        '  image.image_date_annotated, '
        '  image.image_height, image.image_width, '
        '  image_species.species_name, '
        '  image_user_added.user_username, image_user_annotated.user_username '
        'from image '
        'join species as image_species on '
        '  image.image_species_id=image_species.species_id '
        'join user as image_user_added on '
        '  image.image_user_id_added=image_user_added.user_id '
        'join user as image_user_annotated on '
        '  image.image_user_id_annotated=image_user_annotated.user_id '
        'where image.image_id=?',
        species_id_string,
        #'select image_id, image_filepath, image_height, image_width,'
        #' species_name, user_username '
        #'from image, species, user '
        #'where image_id=? and '
        #'image_species_id=species_id and '
        #'image_user_id=user_id',
        #species_id_string,
    )

    (image_id, image_filepath,
        image_date_added, image_date_collected, image_date_annotated,
        image_height, image_width,
        species_name,
        user_username_added, user_username_annotated) = cur.fetchone()

    return json.dumps({
        'status': 'OK',
        'image_id': image_id,
        'image_filepath': image_filepath,
        'image_date_added': image_date_added,
        'image_date_collected': image_date_collected,
        'image_date_annotated': image_date_annotated,
        'image_width': image_width,
        'image_height': image_height,
        'species_name': species_name,
        'username_added': user_username_added,
        'username_annotated': user_username_annotated,
    })


# when user chooses to work on labeling images manually
@app.route('/label')
def label_images():
    cur = g.db.execute(
        'select image_id, image_filepath, species_name '
        'from image, species '
        'where image_species_id = species_id '
        'order by random() limit 1000'  # TODO: just for testing
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
    status_string = str(request.form['status'])
    source_string = str(request.form['source'])
    species_string = str(request.form['species'])

    # map to 0 or 1
    source = ['Algorithm', 'Human'].index(source_string)

    print('  status_string = %s' % status_string)
    print('  source = %d' % source)
    print('  species_string = %s' % species_string)

    if status_string == 'Unannotated':
        cur = g.db.execute(
            'select '
            '  count(image_id) '
            'from image '
            'where'
            '  image_species_id is null '
        )
    elif status_string == 'Annotated':
        if species_string == 'All':
            cur = g.db.execute(
                'select '
                '  count(image_id) '
                'from image '
                'join user as image_user_annotated on'
                '  image.image_user_id_annotated=image_user_annotated.user_id '
                'where'
                '  image_species_id is not null and '
                '  image_user_annotated.user_human=?',
                (source,)
            )
        else:
            cur = g.db.execute(
                'select '
                '  count(image_id) '
                'from image '
                'join user as image_user_annotated on'
                '  image.image_user_id_annotated=image_user_annotated.user_id '
                'where'
                '  image_species_id=('
                '    select '
                '      species_id '
                '    from species '
                '    where '
                '      species_name=?) and'
                '  image_user_annotated.user_human=?',
                (species_string, source)
            )

    result = cur.fetchone()[0]
    print('result = %s' % result)

    model_available = app.config['MODEL'] is not None

    return json.dumps({'count': result, 'model_available': model_available})


@app.route('/review', methods=['POST'])
def review_annotations():
    def get_class_scores(filenames, species):
        if app.config['MODEL'] is not None:
            # tell the model in which order the probabilities are expected
            species_scores = app.config['MODEL'].get_class_scores_filenames(
                filenames, species)

        else:
            uniform = 1. / len(species)
            species_scores = []
            for _ in filenames:
                species_scores.append({
                    sp: uniform for sp in species
                })

        return species_scores

    def get_novelty_scores(filenames, species):
        # TODO: invent a real way to decide when an image should be reviewed
        y_hat = np.random.random(len(filenames))
        return y_hat

    limit_string = str(request.form['limit'])
    status_string = str(request.form['status'])
    source_string = str(request.form['source'])
    species_string = str(request.form['species'])
    probability = float(request.form['probability'])
    novelty = float(request.form['novelty'])

    source = ['Algorithm', 'Human'].index(source_string)

    print('review_annotations')
    print(' limit: %s, status: %s' % (limit_string, status_string))
    print(' probability: %.2f, novelty: %.2f' % (probability, novelty))

    if status_string == 'Unannotated':
        cur = g.db.execute(
            'select'
            '  image.image_id, image.image_filepath, '
            '  image.image_date_added, image.image_date_collected,'
            '  "None",'
            '  image.image_height, image_width, '
            '  "None",'
            '  image_user_added.user_username,'
            '  "None" '
            'from image '
            'join user as image_user_added on'
            '  image.image_user_id_added=image_user_added.user_id '
            'where image.image_species_id is null '
            'limit ?', (limit_string,)
        )
    elif status_string == 'Annotated':
        if species_string == 'All':
            cur = g.db.execute(
                'select'
                '  image.image_id, image.image_filepath, '
                '  image.image_date_added, image.image_date_collected,'
                '  image.image_date_annotated,'
                '  image.image_height, image_width, '
                '  image_species.species_name,'
                '  image_user_added.user_username,'
                '  image_user_annotated.user_username '
                'from image '
                'join species as image_species on '
                '  image.image_species_id=image_species.species_id '
                'join user as image_user_added on'
                '  image.image_user_id_added=image_user_added.user_id '
                'join user as image_user_annotated on'
                '  image.image_user_id_annotated=image_user_annotated.user_id '
                'where '
                '  image.image_species_id is not null and '
                '  image_user_annotated.user_human=? '
                'limit ?', (source, limit_string,)
            )
        else:
            cur = g.db.execute(
                'select'
                '  image.image_id, image.image_filepath, '
                '  image.image_date_added, image.image_date_collected,'
                '  image.image_date_annotated,'
                '  image.image_height, image_width, '
                '  image_species.species_name,'
                '  image_user_added.user_username,'
                '  image_user_annotated.user_username '
                'from image '
                'join species as image_species on '
                '  image.image_species_id=image_species.species_id '
                'join user as image_user_added on'
                '  image.image_user_id_added=image_user_added.user_id '
                'join user as image_user_annotated on'
                '  image.image_user_id_annotated=image_user_annotated.user_id '
                'where '
                '  image.image_species_id=('
                '    select '
                '      species_id '
                '    from species '
                '    where '
                '      species_name=?) and'
                '  image_user_annotated.user_human=? '
                'limit ?', (species_string, source, limit_string,)
            )

    result = cur.fetchall()

    cur = g.db.execute(
        'select species_name from species'
    )
    species = [s[0] for s in cur.fetchall()]
    image_filepaths = [str(x[1]) for x in result]

    # list of dicts: map species name to prob. of that species for this image
    image_predictions_list = get_class_scores(image_filepaths, species)
    novelty_scores = get_novelty_scores(image_filepaths, species)

    review_values, annotate_values = [], []
    for prediction_dict, novelty_score, result_tuple in zip(
            image_predictions_list, novelty_scores, result):
        # unpack the database query
        (image_id, image_filepath,
            image_date_added, image_date_collected, image_date_annotated,
            image_height, image_width,
            species_name, user_added, user_annotated) = result_tuple

        # key-value correspondence is preserved
        species_names = prediction_dict.keys()
        species_probs = prediction_dict.values()

        # sort species names by descending probability
        species_probs_sorted, species_names_sorted = zip(
            *(sorted(zip(
                species_probs, species_names))))

        # this image can be auto-annotated
        if species_probs_sorted[-1] > probability and novelty_score < novelty:
            annotate_values.append((
                strftime('%Y-%m-%d %H:%M:%S'),
                species_names_sorted[-1],
                #app.config['MODELFILE'],
                'convnet',
                image_id,
            ))
        # this image needs to be sent back for review
        else:
            name_prob_tuples = [(name, prob) for name, prob in zip(
                species_names_sorted, species_probs_sorted)][::-1]

            review_values.append({
                'image_id': image_id,
                'image_scores': name_prob_tuples,
                'image_filepath': image_filepath,
                'image_date_added': image_date_added,
                'image_date_collected': image_date_collected,
                'image_date_annotated': image_date_annotated,
                'image_height': image_height,
                'image_width': image_width,
                'species_name': species_name,
                'username_added': user_added,
                'username_annotated': user_annotated,
            })

    cur = g.db.executemany(
        'update image '
        'set '
        'image_date_annotated=?, '
        'image_species_id=('
        '  select species_id from species where species_name=?), '
        'image_user_id_annotated=('
        '  select user_id from user where user_username=?) '
        'where '
        'image_id=?',
        annotate_values
    )

    g.db.commit()

    return json.dumps(review_values)


@app.route('/update', methods=['POST'])
def post_revisions():
    image_id_string = request.form['image_id']
    species_name_string = request.form['species_name']
    # TODO: need to get this from the interface
    username_annotated_string = 'hendrik'
    image_date_annotated = strftime('%Y-%m-%d %H:%M:%S')

    values = (
        species_name_string, username_annotated_string,
        image_date_annotated, image_id_string
    )
    cur = g.db.execute(
        'update image '
        'set '
        'image_species_id=('
        '  select species_id from species where species_name=?), '
        'image_user_id_annotated=('
        '  select user_id from user where user_username=?), '
        'image_date_annotated=? '
        'where image_id=?', values
    )

    g.db.commit()

    if cur.rowcount == 1:
        return json.dumps({
            'status': 'OK',
            'username_annotated': username_annotated_string,
            'image_date_annotated': image_date_annotated
        })
    else:
        return json.dumps({'status': 'ERROR'})


@app.before_first_request
def before_first_request():
    try:
        # TODO: cause the import to fail to accelerate non-learning development
        #from learning_ import Model
        from learning import Model
        conn = connect_db()
        cur = conn.cursor()
        cur.execute('select species_name from species')
        species = []
        for (species_name,) in cur.fetchall():
            species.append(str(species_name))

        cur.close()

        channels = app.config['CHANNELS']
        height, width = app.config['HEIGHT'], app.config['WIDTH']
        app.config['MODEL'] = Model((None, channels, height, width), species)
        app.config['MODEL'].load(join('models', app.config['MODELFILE']))
        app.config['MODEL'].initialize_inference()
    except ImportError:
        warnings.warn('Could not import learning library!')
        app.config['MODEL'] = None


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
    install_secret_key(app, filename='secret_key')

    # first check if we have a custom configuration
    config_file = join(app.instance_path, 'custom.py')
    if not isfile(config_file):
        config_file = join(app.instance_path, 'default.py')
    app.config.from_pyfile(config_file, silent=True)

    app.run(
        host=app.config.get('HOST', '127.0.0.1'),
        port=app.config.get('PORT', 5000)
    )
