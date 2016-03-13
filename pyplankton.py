import sqlite3
from flask import Flask, Response
from flask import request, session, g, redirect, url_for, abort
from flask import send_from_directory
from flask import render_template, flash

from PIL import Image
import StringIO
import json

# configuration
#DATABASE = '/tmp/pyplankton.db'
DATABASE = 'dummy.db'
DEBUG = True
SECRET_KEY = 'development key'
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


@app.route('/', methods=['POST'])
def label_images():
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


@app.route('/overlay', methods=['POST'])
def overlay_image():
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


@app.route('/')
def show_images():
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

    return render_template('show_images.html',
                           #images=map(json.dumps, images),
                           images=images,
                           labels=labels)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('show_images'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('show_images'))
    return render_template('login.html', error=error)


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

if __name__ == '__main__':
    app.run()
