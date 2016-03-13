#!/usr/bin/env python

# This script generates dummy data to make the process of testing
# and developing the plankton annotation interface easier.  The
# random images are written to a local directory.  Random users
# and species are generated and inserted into the dummy database
# along with the dummy image attributes.

import numpy as np
import sqlite3
import string

from PIL import Image

from contextlib import closing
from random import choice, randint

from os import makedirs
from os.path import isdir, join

DATABASE = 'dummy.db'
SCHEMA = 'schema.sql'
DUMMYDIR = 'dummy-data'


def init_db():
    conn = sqlite3.connect(DATABASE)
    with closing(conn) as db:
        with open(SCHEMA, mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()


def generate_dummy_db(num_images=1000, num_species=10, num_users=5):
    def generate_random_string(min_length, max_length):
        return ''.join(
            choice(string.ascii_uppercase) for _ in range(
                randint(min_length, max_length)))

    #  to store the dummy images
    if not isdir(DUMMYDIR):
        print('creating directory %s' % (DUMMYDIR))
        makedirs(DUMMYDIR)

    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    # generate random species names
    dummy_species = []
    for _ in range(num_species):
        dummy_species.append((
            generate_random_string(1, 25),
        ))

    # insert the random species
    cur.executemany(
        'insert into species(species_name) '
        'values(?)', dummy_species
    )
    conn.commit()

    # generate random usernames and passwords
    dummy_users = []
    for _ in range(num_users):
        dummy_users.append((
            generate_random_string(1, 10),
            generate_random_string(1, 10),
        ))

    # insert the random users
    cur.executemany(
        'insert into user(user_username, user_password) '
        'values(?, ?)', dummy_users
    )
    conn.commit()

    # generate random images and write them to disk
    dummy_images = []
    dims = np.random.randint(32, 129, size=(num_images, 2))
    for i, (width, height) in enumerate(dims):
        data = np.random.randint(0, 256, size=(height, width))
        fname = join(DUMMYDIR, '%d.jpg' % i)
        img = Image.fromarray(data.astype(np.uint8))
        img.save(fname)

        dummy_images.append((
            fname,
            height,
            width,
            choice(range(len(dummy_species))),
            choice(range(len(dummy_users))),
        ))

    # insert the random images
    cur.executemany(
        'insert into image('
        'image_filepath, image_height, image_width, '
        'image_species_id, image_user_id) '
        'values(?, ?, ?, ?, ?)',
        dummy_images
    )

    conn.commit()
    cur.close()


def main():
    print('initializing database')
    init_db()

    print('generating dummy data')
    generate_dummy_db(num_images=1000, num_species=25, num_users=5)
    print('done')


if __name__ == '__main__':
    main()
