import sys
import numpy as np
import sqlite3
import warnings
from flask import Flask, Response
from flask import request, session, g, redirect, url_for, abort
from flask import send_from_directory
from flask import render_template
from werkzeug import check_password_hash

from PIL import Image
import StringIO
import json
from time import strftime
from os.path import join, dirname, isdir, isfile

# configuration is done in instance/default.py
app = Flask(__name__)
app.config.from_object(__name__)

MAINTENANCE = False
MAINTENANCE_INFO = ''


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
    if MAINTENANCE:
        return render_template('maintenance.html', error=MAINTENANCE_INFO)
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if MAINTENANCE:
        return render_template('maintenance.html', error=MAINTENANCE_INFO)
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        login_successful = False
        if username and password:
            cur = g.db.execute(
                'select'
                ' user_username, '
                ' user_password '
                'from user where user_username=?', (username,)
            )
            result = cur.fetchone()
            if result is not None:
                user_username, user_password = result
                if check_password_hash(user_password, password):
                    login_successful = True

        if login_successful:
            app.logger.info('login: successful login by user %s' % (
                user_username))
            session['logged_in'] = True
            session['username'] = user_username
            return redirect(url_for('overview'))
        # need to use the provided username here, not the one from the database
        else:
            app.logger.info('login: unsuccessful login by user %s' % (
                username))
            error = 'Invalid username or password'

    return render_template('home.html', error=error)


@app.route('/logout')
def logout():
    if MAINTENANCE:
        return render_template('maintenance.html', error=MAINTENANCE_INFO)
    session.pop('logged_in', None)
    user_username = session.pop('username', None)
    app.logger.info('logout: logout by user %s' % (user_username))
    return render_template('home.html')


@app.route('/overview', methods=['GET'])
def overview():
    if MAINTENANCE:
        return render_template('maintenance.html', error=MAINTENANCE_INFO)
    if not session.get('logged_in'):
        return render_template(
            'home.html', error='You must be logged in to do that')

    cur = g.db.execute(
        'select'
        '  image_family.family_name,'
        '  image_genus.genus_name,'
        '  image_species.species_name,'
        '  count(image.image_id) '
        'from image '
        'left outer join family as image_family on'
        '  image.image_family_id=image_family.family_id '
        'left outer join genus as image_genus on '
        '  image.image_genus_id=image_genus.genus_id '
        'left outer join species as image_species on'
        '  image.image_species_id=image_species.species_id '
        'group by '
        '  image_family.family_id,'
        '  image_genus.genus_id,'
        '  image_species.species_id'
    )

    result = cur.fetchall()
    image_counts = []
    for family_name, genus_name, species_name, image_count in result:
        image_counts.append({
            'family_name': str(family_name),
            'genus_name': str(genus_name),
            'species_name': str(species_name),
            'image_count': image_count,
        })

    cur = g.db.execute(
        'select count(image_id) '
        'from image '
        'where image_family_id is not null'
    )
    family_annotated = cur.fetchone()[0]

    cur = g.db.execute(
        'select count(image_id) '
        'from image '
        'where image_genus_id is not null'
    )
    genus_annotated = cur.fetchone()[0]

    cur = g.db.execute(
        'select count(image_id) '
        'from image '
        'where image_species_id is not null'
    )
    species_annotated = cur.fetchone()[0]

    cur = g.db.execute(
        'select count(image_id) '
        'from image'
    )
    total = cur.fetchone()[0]

    app.logger.info(
        'overview: total = %d' % (total) +
        ', family = %d, genus = %d, species = %d' % (
            family_annotated, genus_annotated, species_annotated))
    return render_template('overview.html',
                           family_annotated=family_annotated,
                           genus_annotated=genus_annotated,
                           species_annotated=species_annotated,
                           total=total,
                           image_counts=image_counts)


@app.route('/overview/update', methods=['POST'])
def overview_update():
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

    app.logger.info('overview_update: set species %s to confusable = %s, updated = %d' % (
        species_name_string, species_confusable, cur.rowcount))
    return json.dumps({'status': 'OK'})


# when user chooses a new species for a set of images
@app.route('/update_labels', methods=['POST'])
def post_labels():
    if not session.get('logged_in'):
        return json.dump({'status': 'ERROR', 'rows_updated': 0})

    image_id_string = request.form['image_id']
    image_id_list = image_id_string.split(', ')

    # get the id and which field to set
    species_id_string = request.form['species_id']
    genus_id_string = request.form['genus_id']
    family_id_string = request.form['family_id']
    #type_string = request.form['type']

    current_user = session['username']
    cur = g.db.execute(
        'select user_id from user where user_username=?',
        (current_user,)
    )
    (current_user_id,) = cur.fetchone()

    current_time = strftime('%Y-%m-%d %H:%M:%S')

    # nulls from javascript arrive as empty strings...
    if species_id_string == '':
        image_species_id = None
        image_date_species_annotated = None
        image_user_id_species_annotated = None
    else:
        assert species_id_string.isdigit(), 'species_id_string must be an int'
        image_species_id = int(species_id_string)
        image_date_species_annotated = current_time
        image_user_id_species_annotated = current_user_id

    if genus_id_string == '':
        image_genus_id = None
        image_date_genus_annotated = None
        image_user_id_genus_annotated = None
    else:
        assert genus_id_string.isdigit(), 'genus_id_string must be an int'
        image_genus_id = int(genus_id_string)
        image_date_genus_annotated = current_time
        image_user_id_genus_annotated = current_user_id

    if family_id_string == '':
        image_family_id = None
        image_date_family_annotated = None
        image_user_id_family_annotated = None
    else:
        assert family_id_string.isdigit(), 'family_id_string must be an int'
        image_family_id = int(family_id_string)
        image_date_family_annotated = current_time
        image_user_id_family_annotated = current_user_id

    values = []
    for image_id in image_id_list:
        values.append((
            image_family_id,
            image_genus_id,
            image_species_id,
            image_date_family_annotated,
            image_date_genus_annotated,
            image_date_species_annotated,
            image_user_id_family_annotated,
            image_user_id_genus_annotated,
            image_user_id_species_annotated,
            image_id,
        ))

    cur = g.db.executemany(
        'update image '
        'set '
        '  image_family_id=?,'
        '  image_genus_id=?,'
        '  image_species_id=?,'
        '  image_date_family_annotated=?,'
        '  image_date_genus_annotated=?,'
        '  image_date_species_annotated=?,'
        '  image_user_id_family_annotated=?,'
        '  image_user_id_genus_annotated=?,'
        '  image_user_id_species_annotated=? '
        'where '
        'image_id=?',
        values,
    )

    g.db.commit()

    app.logger.info('post_labels: %d images to species_id %s' % (
        len(values), species_id_string))
    # a null in javascript is passed here as an empty string
    info_list = ['  image_id %s set to' % (value[9]) +
                 ' family_id %s on %s by %s' % (value[0], value[3], value[6]) +
                 ' genus_id %s on %s by %s' % (value[1], value[4], value[7]) +
                 ' species_id %s on %s by %s' % (value[2], value[5], value[8])
                 for value in values]
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
        '  image.image_date_family_annotated,'
        '  image.image_date_genus_annotated,'
        '  image.image_date_species_annotated,'
        '  image.image_height, image.image_width, '
        '  image_family.family_name,'
        '  image_family.family_confusable,'
        '  image_genus.genus_name,'
        '  image_genus.genus_confusable,'
        '  image_species.species_name,'
        '  image_species.species_confusable,'
        '  image_user_added.user_username, '
        '  image_user_family_annotated.user_username,'
        '  image_user_genus_annotated.user_username,'
        '  image_user_species_annotated.user_username '
        'from image '
        'join user as image_user_added on '
        '  image.image_user_id_added=image_user_added.user_id '
        'left outer join family as image_family on '
        '  image.image_family_id=image_family.family_id '
        'left outer join genus as image_genus on '
        '  image.image_genus_id=image_genus.genus_id '
        'left outer join species as image_species on '
        '  image.image_species_id=image_species.species_id '
        'left outer join user as image_user_family_annotated on '
        '  image.image_user_id_family_annotated=image_user_family_annotated.user_id '
        'left outer join user as image_user_genus_annotated on '
        '  image.image_user_id_genus_annotated=image_user_genus_annotated.user_id '
        'left outer join user as image_user_species_annotated on '
        '  image.image_user_id_species_annotated=image_user_species_annotated.user_id '
        'where '
        '  image.image_id=?',
        (image_id_string,)
    )

    (image_id, image_filepath,
        image_date_added, image_date_collected,
        image_date_family_annotated,
        image_date_genus_annotated,
        image_date_species_annotated,
        image_height, image_width,
        family_name, family_confusable,
        genus_name, genus_confusable,
        species_name, species_confusable,
        user_username_added,
        user_username_family_annotated,
        user_username_genus_annotated,
        user_username_species_annotated) = cur.fetchone()

    family_confusable_typed = bool(family_confusable) if isinstance(family_confusable, int) else family_confusable
    genus_confusable_typed = bool(genus_confusable) if isinstance(genus_confusable, int) else genus_confusable
    species_confusable_typed = bool(species_confusable) if isinstance(species_confusable, int) else species_confusable
    app.logger.info('post_overlay: image_id = %s' % (image_id_string))
    return json.dumps({
        'status': 'OK',
        'image_id': image_id,
        'image_filepath': image_filepath,
        'image_date_added': image_date_added,
        'image_date_collected': image_date_collected,
        'image_date_family_annotated': str(image_date_family_annotated),
        'image_date_genus_annotated': str(image_date_genus_annotated),
        'image_date_species_annotated': str(image_date_species_annotated),
        'image_width': image_width,
        'image_height': image_height,
        'family_name': str(family_name),
        'family_confusable': str(family_confusable_typed),
        'genus_name': str(genus_name),
        'genus_confusable': str(genus_confusable_typed),
        'species_name': str(species_name),
        'species_confusable': str(species_confusable_typed),
        'username_added': user_username_added,
        'username_family_annotated': str(user_username_family_annotated),
        'username_genus_annotated': str(user_username_genus_annotated),
        'username_species_annotated': str(user_username_species_annotated),
    })


# when user chooses to work on labeling images manually
@app.route('/label')
def label_images():
    if MAINTENANCE:
        return render_template('maintenance.html', error=MAINTENANCE_INFO)
    if not session.get('logged_in'):
        return render_template(
            'home.html', error='You must be logged in to do that')
    cur = g.db.execute(
        'select'
        '  family.family_id, family.family_name, '
        '  family_genus.genus_id, family_genus.genus_name, '
        '  genus_species.species_id, genus_species.species_name '
        'from family '
        'left outer join genus as family_genus on '
        '  family.family_id=family_genus.genus_family_id '
        'left outer join species as genus_species on '
        '  family_genus.genus_id=genus_species.species_genus_id '
        'order by family_name'
    )

    result = cur.fetchall()
    taxonomy_dict = {}
    for result_tuple in result:
        (family_id, family_name,
            genus_id, genus_name,
            species_id, species_name) = result_tuple
        species_name = None if species_name is None else str(species_name)
        genus_name = None if genus_name is None else str(genus_name)
        family_name = None if family_name is None else str(family_name)
        if family_id not in taxonomy_dict:
            taxonomy_dict[family_id] = {
                'family_id': family_id,
                'family_name': family_name,
                'genus_list': {genus_id: {
                    'genus_id': genus_id,
                    'genus_name': genus_name,
                    'species_list': [{
                        'species_id': species_id,
                        'species_name': species_name,
                    }],
                }},
            }
        else:
            if genus_id not in taxonomy_dict[family_id]['genus_list']:
                taxonomy_dict[family_id]['genus_list'][genus_id] = {
                    'genus_id': genus_id,
                    'genus_name': genus_name,
                    'species_list': [{
                        'species_id': species_id,
                        'species_name': species_name,
                    }],
                }
            else:
                taxonomy_dict[family_id]['genus_list'][genus_id]['species_list'].append({
                    'species_id': species_id,
                    'species_name': species_name,
                })

    family_list = []
    for family_id in taxonomy_dict:
        genus_list = taxonomy_dict[family_id]['genus_list'].values()
        # sort the genera alphabetically by name
        genus_list_alpha = sorted(genus_list, key=lambda k: k['genus_name'])
        taxonomy_dict[family_id]['genus_list'] = genus_list_alpha
        family_list.append(taxonomy_dict[family_id])

    # sort the families alphabetically by name
    family_list_alpha = sorted(family_list, key=lambda k: k['family_name'])
    family_list_alpha.append({'family_id': None, 'family_name': 'None'})

    cur = g.db.execute('select user_id, user_username from user')
    result = cur.fetchall()
    user_list = []
    for user_id, user_name in result:
        user_list.append({
            'user_id': user_id,
            'user_username': user_name
        })

    #app.logger.info('label_images: %d species' % (len(species)))
    return render_template('label_images.html',
                           families=family_list_alpha,
                           users=user_list)


@app.route('/label', methods=['POST'])
def begin_label():
    if not session.get('logged_in'):
        return render_template(
            'home.html', error='You must be logged in to do that')
    limit_string = str(request.form['limit'])
    source_string = str(request.form['source'])
    family_string = str(request.form['family'])
    genus_string = str(request.form['genus'])
    species_string = str(request.form['species'])
    order_string = str(request.form['order'])

    #source = ['Algorithm', 'Human'].index(source_string)

    app.logger.info(
        'begin_label:'
        ' limit = %s, source = %s, ' % (
            limit_string, source_string) +
        'family = %s, genus = %s, species = %s' % (
            family_string, genus_string, species_string)
    )

    most_specific_rank = None  # need to know which table to query for the user
    values = []
    if family_string == genus_string == species_string == 'All':
        where_clause = (
            'where'
            ' image_family.family_id is not null and'
            ' image_genus.genus_id is not null and '
            ' image_species.species_id is not null')
        most_specific_rank = 'species'
    elif family_string == genus_string == 'All':
        assert species_string == 'None'
        where_clause = (
            'where'
            ' image_family.family_id is not null and'
            ' image_genus.genus_id is not null and '
            ' image_species.species_id is null')
        most_specific_rank = 'genus'
    elif family_string == 'All':
        assert genus_string == species_string == 'None'
        where_clause = (
            'where'
            ' image_family.family_id is not null and'
            ' image_genus.genus_id is null and '
            ' image_species.species_id is null')
        most_specific_rank = 'family'
    elif family_string == 'None':
        assert genus_string == species_string == 'None'
        # in practice if family_id is null the others should be too
        where_clause = (
            'where'
            ' image_family.family_id is null and'
            ' image_genus.genus_id is null and'
            ' image_species.species_id is null')
    else:
        assert family_string.isdigit(), (
            'family_id must be an int to query')
        family_id = family_string
        values += [family_id]
        most_specific_rank = 'family'
        if genus_string == 'All':
            where_clause = (
                'where'
                ' image_family.family_id=? and'
                ' image_genus.genus_id is not null')
            most_specific_rank = 'genus'
        elif genus_string == 'None':
            assert species_string == 'None'
            where_clause = (
                'where'
                ' image_family.family_id=? and'
                ' image_genus.genus_id is null')
        else:
            assert genus_string.isdigit(), (
                'genus_id must be an int to query')
            genus_id = genus_string
            values += [genus_id]
            most_specific_rank = 'genus'
            if species_string == 'All':
                where_clause = (
                    'where'
                    ' image_family.family_id=? and'
                    ' image_genus.genus_id=? and '
                    ' image_species.species_id is not null')
                most_specific_rank = 'species'
            elif species_string == 'None':
                where_clause = (
                    'where'
                    ' image_family.family_id=? and'
                    ' image_genus.genus_id=? and '
                    ' image_species.species_id is null')
            else:
                assert species_string.isdigit(), (
                    'species_id must be an int to query')
                species_id = species_string
                values += [species_id]
                where_clause = (
                    'where'
                    ' image_family.family_id=? and'
                    ' image_genus.genus_id=? and'
                    ' image_species.species_id=?')
                most_specific_rank = 'species'

    # get images where the most specific level of classification
    # was made by this user
    if source_string == 'Humans & Algorithms':
        pass
    elif source_string == 'Algorithms':
        assert most_specific_rank is not None
        where_clause += (
            ' and image_user_%s_annotated.user_human=0' % most_specific_rank)
    elif source_string == 'Humans only':
        assert most_specific_rank is not None
        where_clause += (
            ' and image_user_%s_annotated.user_human=1' % most_specific_rank)
    else:
        user_id = source_string
        assert most_specific_rank is not None
        where_clause += (
            ' and image_user_%s_annotated.user_id=?' % most_specific_rank)
        values += [user_id]

    if order_string == 'Image Similarity':
        # image similarity is only implemented for unannotated images
        if family_string == 'None':
            where_clause += (' and image.image_cluster_id=?')
            order_by = 'order by image.image_cluster_dist asc'
            values += [np.random.randint(0, 60)]  # TODO: fix max. value
        else:
            order_by = ''
    elif order_string == 'Date Added':
        order_by = 'order by image.image_date_added'
    elif order_string == 'Date Collected':
        order_by = 'order by image.image_date_collected'
    else:
        assert False, '%s is not a valid ordering' % (order_string)

    # the limit is always last
    values += [limit_string]

    cur = g.db.execute(
        'select'
        '  image.image_id, image.image_filepath, '
        '  image.image_date_added, image.image_date_collected, '
        '  image.image_date_family_annotated,'
        '  image.image_date_genus_annotated,'
        '  image.image_date_species_annotated,'
        '  image.image_height, image.image_width, '
        '  image_family.family_name,'
        '  image_family.family_confusable,'
        '  image_genus.genus_name,'
        '  image_genus.genus_confusable,'
        '  image_species.species_name,'
        '  image_species.species_confusable,'
        '  image_user_added.user_username,'
        '  image_user_family_annotated.user_username '
        'from image '
        'join user as image_user_added on '
        '  image.image_user_id_added=image_user_added.user_id '
        'left outer join family as image_family on '
        '  image.image_family_id=image_family.family_id '
        'left outer join genus as image_genus on '
        '  image.image_genus_id=image_genus.genus_id '
        'left outer join species as image_species on '
        '  image.image_species_id=image_species.species_id '
        'left outer join user as image_user_family_annotated on '
        '  image.image_user_id_family_annotated'
        '    =image_user_family_annotated.user_id '
        'left outer join user as image_user_genus_annotated on '
        '  image.image_user_id_genus_annotated'
        '    =image_user_genus_annotated.user_id '
        'left outer join user as image_user_species_annotated on '
        '  image.image_user_id_species_annotated'
        '    =image_user_species_annotated.user_id '
        + where_clause + ' '
        + order_by + ' '
        'limit ?', values
    )

    result = cur.fetchall()

    images = []
    width, height = 95, 95
    for result_tuple in result:
        (image_id, image_filepath,
            image_date_added, image_date_collected,
            image_date_family_annotated,
            image_date_genus_annotated,
            image_date_species_annotated,
            image_height, image_width,
            family_name, family_confusable,
            genus_name, genus_confusable,
            species_name, species_confusable,
            user_added, user_annotated) = result_tuple

        # convert to string representation of boolean or leave as N/A
        family_confusable_typed = bool(family_confusable) if isinstance(family_confusable, int) else family_confusable
        genus_confusable_typed = bool(genus_confusable) if isinstance(genus_confusable, int) else genus_confusable
        species_confusable_typed = bool(species_confusable) if isinstance(species_confusable, int) else species_confusable

        species_name = None if species_name is None else str(species_name)
        genus_name = None if genus_name is None else str(genus_name)
        family_name = None if family_name is None else str(family_name)
        images.append({
            'image_id': image_id,
            'image_filepath': image_filepath,
            'image_date_added': image_date_added,
            'image_date_collected': image_date_collected,
            'image_date_family_annotated': str(image_date_family_annotated),
            'image_date_genus_annotated': str(image_date_genus_annotated),
            'image_date_species_annotated': str(image_date_species_annotated),
            'image_height': image_height,
            'image_width': image_width,
            'family_name': family_name,
            'family_confusable': family_confusable_typed,
            'genus_name': genus_name,
            'genus_confusable': genus_confusable_typed,
            'species_name': species_name,
            'species_confusable': species_confusable_typed,
            'username_added': user_added,
            'username_annotated': str(user_annotated),
            'width': width,
            'height': height,
            'src': image_filepath,
        })

    app.logger.info('begin_label: return %d images' % (len(images)))
    return json.dumps(images)


# when user chooses to review species
@app.route('/review')
def review_images():
    if MAINTENANCE:
        return render_template('maintenance.html', error=MAINTENANCE_INFO)
    if not session.get('logged_in'):
        return render_template(
            'home.html', error='You must be logged in to do that')
    # delete else-clause when review interface is available again
    else:
        return render_template('home.html', error=None)

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

    cur = g.db.execute(
        'select species_name from species where species_confusable=1'
    )
    confusables_species_names = [s for (s,) in cur.fetchall()]

    image_filepaths = [str(x[1]) for x in result]
    app.logger.info('begin_review: filepaths: %d species: %d' % (
        len(image_filepaths), len(species)))

    # list of dicts: map species name to prob. of that species for this image
    image_predictions_list = get_class_scores(image_filepaths, species)
    novelty_scores = get_novelty_scores(image_filepaths, species)

    review_values, annotate_values = [], []
    auto_annotate_values = []
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

        # this image can be auto-annotated if:
        # 1. the largest softmax output exceeds the given prob.
        # 2. the novelty score does not exceed the given prob.
        # 3. the species provided by the model has not been set as confusable
        target_species_prob = species_probs_sorted[-1]  # max. softmax output
        target_species_name = species_names_sorted[-1]  # corresponding species
        target_species_confusable = target_species_name in confusables_species_names
        if target_species_prob > probability \
                and novelty_score < novelty \
                and not target_species_confusable:

            annotate_values.append((
                strftime('%Y-%m-%d %H:%M:%S'),
                target_species_name,
                #app.config['MODELFILE'],
                'convnet',
                image_id,
            ))

            auto_annotate_values.append((
                target_species_prob,
                novelty_score,
                target_species_confusable,
            ))
        # this image needs to be sent back for review
        else:
            name_prob_tuples = [(name, prob) for name, prob in zip(
                species_names_sorted, species_probs_sorted)][::-1]

            # convert to string representation of boolean or leave as None
            species_confusable_typed = bool(species_confusable) if isinstance(species_confusable, int) else species_confusable

            review_values.append({
                'image_id': image_id,
                'image_scores': name_prob_tuples,
                'image_filepath': image_filepath,
                'image_date_added': image_date_added,
                'image_date_collected': image_date_collected,
                'image_date_annotated': str(image_date_annotated),
                'image_height': image_height,
                'image_width': image_width,
                'species_name': str(species_name),
                'species_confusable': str(species_confusable_typed),
                'username_added': user_added,
                'username_annotated': str(user_annotated),
            })

    app.logger.info('begin_review: %d images for review, %d auto-annotated' % (
        len(review_values), len(annotate_values)))

    # log info regarding the images that were auto-annotated
    app.logger.info('begin_review: min. prob. = %.2f, max. novel. = %.2f, confusable species:\n%s' % (probability, novelty, '\n  '.join(confusables_species_names)))
    info_list = []
    for annot, auto in zip(annotate_values, auto_annotate_values):
        info_list.append(
            '  image_id %d set to species %s by %s on %s\n' % (
                annot[3], annot[1], annot[2], annot[0]
            ) +
            '    (prob. = %.2f, novel. = %.2f, conf. = %s)' % (
                auto[0], auto[1], auto[2]
            )
        )
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

    username_annotated_string = session['username']
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
        from learning_ import Model
        #from learning import Model
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
