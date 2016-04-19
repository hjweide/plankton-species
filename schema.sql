drop table if exists user;
drop table if exists image;
drop table if exists family;
drop table if exists genus;
drop table if exists species;
pragma foreign_keys = ON;
create table user (
  user_id integer primary key autoincrement,
  user_human boolean not null check (user_human in (0, 1)), -- false if this user is an algorithm
  user_username text not null unique,
  user_password text not null
);
create table family (
  family_id integer primary key autoincrement,
  family_name text not null unique,
  family_confusable boolean not null check (family_confusable in (0, 1))
);
create table genus (
  genus_id integer primary key autoincrement,
  genus_name text not null unique,
  genus_confusable boolean not null check (genus_confusable in (0, 1)),
  genus_family_id integer not null,
  foreign key(genus_family_id) references family(family_id)
);
create table species (
  species_id integer primary key autoincrement,
  species_name text not null unique,
  species_confusable boolean not null check (species_confusable in (0, 1)),
  species_genus_id integer not null,
  foreign key(species_genus_id) references genus(genus_id)
);
create table image (
  image_id integer primary key autoincrement,
  image_filepath text not null unique,

  image_user_id_added integer,
  image_user_id_family_annotated integer,
  image_user_id_genus_annotated integer,
  image_user_id_species_annotated integer,

  image_date_collected datetime, -- the image timestamp
  image_date_added datetime, -- the timestamp when the image was added to the db

  image_date_family_annotated datetime, -- the timestamp when a family label was assigned
  image_date_genus_annotated datetime, -- the timestamp when a genus label was assigned
  image_date_species_annotated datetime, -- the timestamp when a species label was assigned

  image_height integer not null,
  image_width integer not null,

  image_family_id integer,
  image_genus_id integer,
  image_species_id integer,

  image_cluster_id integer,
  image_cluster_dist real,

  foreign key(image_family_id) references family(family_id),
  foreign key(image_genus_id) references genus(genus_id),
  foreign key(image_species_id) references species(species_id),

  foreign key(image_user_id_added) references user(user_id),
  foreign key(image_user_id_family_annotated) references user(user_id),
  foreign key(image_user_id_genus_annotated) references user(user_id),
  foreign key(image_user_id_species_annotated) references user(user_id),
  -- if the species is not null, we need to know the user that set it
  -- if the species is null, no user could have set it
  check (
    (image_species_id is null and image_user_id_species_annotated is null and image_date_species_annotated is null) or
    (image_species_id is not null and image_user_id_species_annotated is not null and image_date_species_annotated is not null)
  ),
  check (
    (image_genus_id is null and image_user_id_genus_annotated is null and image_date_genus_annotated is null) or
    (image_genus_id is not null and image_user_id_genus_annotated is not null and image_date_genus_annotated is not null)
  ),
  check (
    (image_family_id is null and image_user_id_family_annotated is null and image_date_family_annotated is null) or
    (image_family_id is not null and image_user_id_family_annotated is not null and image_date_family_annotated is not null)
  ),
  -- check that more specific classifications don't exist without a more general one
  check (
    (image_family_id is null and image_genus_id is null and image_species_id is null) or 
    (image_family_id is not null and image_genus_id is null and image_species_id is null) or 
    (image_family_id is not null and image_genus_id is not null and image_species_id is null) or 
    (image_family_id is not null and image_genus_id is not null and image_species_id is not null)
  ),

  -- an image assigned to a cluster must also have a distance to that cluster
  check(
    (image_cluster_id is null and image_cluster_dist is null) or
    (image_cluster_id is not null and image_cluster_dist is not null)
  )
);
