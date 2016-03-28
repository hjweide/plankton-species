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

    app.logger.info('home: annotated = %d, total = %d, len(counts) = %d' % (
        annotated, total, len(species_counts)))
    return render_template('home.html',
                           annotated=annotated,
                           total=total,
                           species_counts=species_counts)


@app.route('/home/update', methods=['POST'])
def home_update():
    species_name_string = request.form['species_name']
    species_confusable_string = request.form['species_confusable']
    species_confusable = 1 if species_confusable_string == 'true' else 0
    cur = g.db.execute(
        'update'
        '  species '
        'set '
        '  species_confusable=? '
        'where'
        '  species_name=?',
        (species_confusable, species_name_string)
    )

    g.db.commit()

    app.logger.info('home: set species %s to confusable = %s, updated = %d' % (
        species_name_string, species_confusable, cur.rowcount))
    return json.dumps({'status': 'OK'})


# when user chooses a new species for a set of images
@app.route('/update_labels', methods=['POST'])
def post_labels():
    image_id_string = request.form['image_id']
    species_id_string = request.form['species_id']
    image_id_list = image_id_string.split(', ')

    # TODO: need to get this from the interface
    username_annotated_string = 'hendrik'
    image_date_annotated = strftime('%Y-%m-%d %H:%M:%S')

    values = []
    for image_id in image_id_list:
        values.append((
            species_id_string,
            image_date_annotated,
            username_annotated_string,
            image_id,
        ))

    cur = g.db.executemany(
        'update image '
        'set '
        '  image_species_id=?, '
        '  image_date_annotated=?, '
        '  image_user_id_annotated=('
        '    select user_id from user where user_username=?) '
        'where '
        'image_id=?',
        values
    )

    g.db.commit()

    app.logger.info('post_labels: %d images to species_id %s' % (
        len(values), species_id_string))
    info_list = ['  image_id %s set to species_id %s by %s on %s' % (
        value[3], value[0], value[2], value[1]) for value in values]
    app.logger.info('post_labels:\n%s' % ('\n'.join(info_list)))
    return json.dumps({'status': 'OK', 'rows_updated': cur.rowcount})


# called to overlay the native resolution of the selected image
@app.route('/overlay', methods=['POST'])
def post_overlay():
    image_id_string = request.form['image_id']
    cur = g.db.execute(
        'select'
        '  image.image_id, image.image_filepath, '
        '  image.image_date_added, image.image_date_collected, '
        '  image.image_date_annotated,'
        '  image.image_height, image.image_width, '
        '  image_species.species_name,'
        '  image_species.species_confusable,'
        '  image_user_added.user_username, '
        '  image_user_annotated.user_username '
        'from image '
        'join user as image_user_added on '
        '  image.image_user_id_added=image_user_added.user_id '
        'left outer join species as image_species on '
        '  image.image_species_id=image_species.species_id '
        'left outer join user as image_user_annotated on '
        '  image.image_user_id_annotated=image_user_annotated.user_id '
        'where '
        '  image.image_id=?',
        (image_id_string,)
    )

    (image_id, image_filepath,
        image_date_added, image_date_collected, image_date_annotated,
        image_height, image_width,
        species_name, species_confusable,
        user_username_added, user_username_annotated) = cur.fetchone()

    species_confusable_typed = bool(species_confusable) if isinstance(species_confusable, int) else species_confusable
    app.logger.info('post_overlay: image_id = %s' % (image_id_string))
    return json.dumps({
        'status': 'OK',
        'image_id': image_id,
        'image_filepath': image_filepath,
        'image_date_added': image_date_added,
        'image_date_collected': image_date_collected,
        'image_date_annotated': str(image_date_annotated),
        'image_width': image_width,
        'image_height': image_height,
        'species_name': str(species_name),
        'species_confusable': str(species_confusable_typed),
        'username_added': user_username_added,
        'username_annotated': str(user_username_annotated),
    })


# when user chooses to work on labeling images manually
@app.route('/label')
def label_images():
    cur = g.db.execute(
        'select species_id, species_name from species order by species_id'
    )
    result = cur.fetchall()
    species = []
    for (species_id, species_name,) in result:
        species.append({
            'species_id': species_id,
            'species_name': species_name,
        })

    app.logger.info('label_images: %d species' % (len(species)))
    return render_template('label_images.html',
                           species=species)


@app.route('/label', methods=['POST'])
def begin_label():
    limit_string = str(request.form['limit'])
    status_string = str(request.form['status'])
    source_string = str(request.form['source'])
    species_string = str(request.form['species'])

    source = ['Algorithm', 'Human'].index(source_string)

    app.logger.info('begin_label:'
                    ' limit = %s, status = %s, source = %s, species = %s' % (
                        limit_string, status_string, source_string,
                        species_string))

    if status_string == 'Unannotated':
        where_clause = ('where'
                        '  image.image_species_id is null ')
        values = (limit_string,)
    elif status_string == 'Annotated':
        if species_string == 'All':
            where_clause = ('where'
                            '  image.image_species_id is not null and '
                            '  image_user_annotated.user_human=? ')
            values = (source, limit_string)
        else:
            where_clause = ('where'
                            '  image.image_species_id=('
                            '    select '
                            '      species_id '
                            '    from species '
                            '    where '
                            '      species_name=?) and'
                            '  image_user_annotated.user_human=? ')
            values = (species_string, source, limit_string)

    cur = g.db.execute(
        'select'
        '  image.image_id, image.image_filepath, '
        '  image.image_date_added, image.image_date_collected, '
        '  image.image_date_annotated,'
        '  image.image_height, image.image_width, '
        '  image_species.species_name,'
        '  image_species.species_confusable,'
        '  image_user_added.user_username, '
        '  image_user_annotated.user_username '
        'from image '
        'join user as image_user_added on '
        '  image.image_user_id_added=image_user_added.user_id '
        'left outer join species as image_species on '
        '  image.image_species_id=image_species.species_id '
        'left outer join user as image_user_annotated on '
        '  image.image_user_id_annotated=image_user_annotated.user_id '
        + where_clause +
        'limit ?', values
    )

    result = cur.fetchall()

    images = []
    width, height = 95, 95
    for result_tuple in result:
        (image_id, image_filepath,
            image_date_added, image_date_collected, image_date_annotated,
            image_height, image_width,
            species_name, species_confusable,
            user_added, user_annotated) = result_tuple

        # convert to string representation of boolean or leave as N/A
        species_confusable_typed = bool(species_confusable) if isinstance(species_confusable, int) else species_confusable

        images.append({
            'image_id': image_id,
            'image_filepath': image_filepath,
            'image_date_added': image_date_added,
            'image_date_collected': image_date_collected,
            'image_date_annotated': str(image_date_annotated),
            'image_height': image_height,
            'image_width': image_width,
            'species_name': str(species_name),
            'species_confusable': species_confusable_typed,
            'username_added': user_added,
            'username_annotated': str(user_annotated),
            'width': width,
            'height': height,
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

    app.logger.info('begin_label: return %d images and %d species' % (
        len(images), len(labels)))
    return json.dumps(images)


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

    app.logger.info('review_images: %d species' % (len(species)))
    return render_template('review_images.html',
                           species=species)


# prepares to query the database for images to be reviewed
@app.route('/prepare', methods=['POST'])
def prepare_review():
    status_string = str(request.form['status'])
    source_string = str(request.form['source'])
    species_string = str(request.form['species'])

    # map to 0 or 1
    source = ['Algorithm', 'Human'].index(source_string)

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

    model_available = app.config['MODEL'] is not None

    if not model_available:
        app.logger.warning('prepare_review: model_available = %s' % (
            model_available))

    app.logger.info(
        'prepare_review:'
        'status = %s, source = %s, species = %s, model_available = %s' % (
            status_string, source_string, species_string, model_available))
    return json.dumps({'count': result, 'model_available': model_available})


@app.route('/review', methods=['POST'])
def begin_review():
    def get_class_scores(filenames, species):
        app.logger.info('get_class_scores: filenames: %d, species: %d' % (
            len(filenames), len(species)))
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

    app.logger.info('begin_review:'
                    ' limit = %s, status = %s, source = %s, species = %s, '
                    'probability = %.2f, novelty = %.2f' % (
                        limit_string, status_string, source_string,
                        species_string, probability, novelty))

    if status_string == 'Unannotated':
        where_clause = 'where image.image_species_id is null '
        values = (limit_string,)
    elif status_string == 'Annotated':
        if species_string == 'All':
            where_clause = ('where'
                            '  image.image_species_id is not null and '
                            '  image_user_annotated.user_human=? ')
            values = (source, limit_string)
        else:
            where_clause = ('where'
                            '  image.image_species_id=('
                            '    select '
                            '      species_id '
                            '    from species '
                            '    where '
                            '      species_name=?) and'
                            '  image_user_annotated.user_human=? ')
            values = (species_string, source, limit_string)

    cur = g.db.execute(
        'select'
        '  image.image_id, image.image_filepath, '
        '  image.image_date_added, image.image_date_collected, '
        '  image.image_date_annotated,'
        '  image.image_height, image.image_width, '
        '  image_species.species_name,'
        '  image_species.species_confusable,'
        '  image_user_added.user_username, '
        '  image_user_annotated.user_username '
        'from image '
        'join user as image_user_added on '
        '  image.image_user_id_added=image_user_added.user_id '
        'left outer join species as image_species on '
        '  image.image_species_id=image_species.species_id '
        'left outer join user as image_user_annotated on '
        '  image.image_user_id_annotated=image_user_annotated.user_id '
        + where_clause +
        'limit ?', values
    )

    result = cur.fetchall()

    cur = g.db.execute(
        'select species_name from species'
    )
    species = [s[0] for s in cur.fetchall()]
    image_filepaths = [str(x[1]) for x in result]
    app.logger.info('begin_review: filepaths: %d species: %d' % (
        len(image_filepaths), len(species)))

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
            species_name, species_confusable,
            user_added, user_annotated) = result_tuple

        # key-value correspondence is preserved
        species_names = prediction_dict.keys()
        species_probs = prediction_dict.values()

        # sort species names by descending probability
        species_probs_sorted, species_names_sorted = zip(
            *(sorted(zip(
                species_probs, species_names))))

        # this image can be auto-annotated
        if species_probs_sorted[-1] > probability and\
                novelty_score < novelty:

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

            # convert to string representation of boolean or leave as N/A
            species_confusable_typed = bool(species_confusable) if isinstance(species_confusable, int) else species_confusable

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
                'species_confusable': species_confusable_typed,
                'username_added': user_added,
                'username_annotated': user_annotated,
            })

    app.logger.info('begin_review: %d images for review, %d auto-annotated' % (
        len(review_values), len(annotate_values)))
    # check that the ordering of annot matches the annotate_values above
    info_list = ['  image_id %d set to species %s by %s on %s' % (
        annot[3], annot[1], annot[2], annot[0]) for annot in annotate_values]
    app.logger.info('begin_review:\n%s' % ('\n'.join(info_list)))

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
    app.logger.info('begin_review: auto-annotated updated %d' % (
        cur.rowcount))

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

    app.logger.info('post_revisions: '
                    'image_id %s, species_name %s, username_annotated %s, '
                    'image_date_annotated %s, updated %s' % (
                        image_id_string, species_name_string,
                        username_annotated_string, image_date_annotated,
                        cur.rowcount))
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
        batch_size = None
        app.logger.info('before_first_request: initializing model with '
                        'input_shape = (%r, %d, %d, %d) and %d species' % (
                            batch_size, channels, height, width, len(species)))
        app.config['MODEL'] = Model(
            (batch_size, channels, height, width), species)
        app.logger.info('before_first_request: loading model configuration '
                        'from %s' % (
                            app.config['MODELFILE']))
        app.config['MODEL'].load(join('models', app.config['MODELFILE']))
        app.config['MODEL'].initialize_inference()
    except ImportError:
        app.logger.warn('Could not import learning library!')
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


if __name__ == '__main__':
    # first check if we have a custom configuration
    config_file = join(app.instance_path, 'custom.py')
    if not isfile(config_file):
        config_file = join(app.instance_path, 'default.py')
    app.config.from_pyfile(config_file, silent=True)

    import logging
    file_handler = logging.FileHandler('debug_logs')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(pathname)s:%(lineno)d]'
    ))

    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.run(
        host=app.config.get('HOST', '127.0.0.1'),
        port=app.config.get('PORT', 5000)
    )
