drop table if exists user;
drop table if exists image;
drop table if exists species;
pragma foreign_keys = ON;
create table user (
  user_id integer primary key autoincrement,
  user_human boolean not null check (user_human in (0, 1)), -- false if this user is an algorithm
  user_username text not null unique,
  user_password text not null
);
create table species (
  species_id integer primary key autoincrement,
  species_name text not null unique,
  species_confusable boolean not null check (species_confusable in (0, 1))
);
create table image (
  image_id integer primary key autoincrement,
  image_filepath text not null unique,
  image_user_id_added integer,
  image_user_id_annotated integer, -- only true when an algorithm sets the species without human supervision
  image_date_collected datetime, -- the image timestamp
  image_date_added datetime, -- the timestamp when the image was added to the db
  image_date_annotated datetime, -- the timestamp when a species label was assigned
  image_height integer not null,
  image_width integer not null,
  image_species_id integer,
  FOREIGN KEY(image_species_id) REFERENCES species(species_id),
  FOREIGN KEY(image_user_id_added) REFERENCES user(user_id),
  FOREIGN KEY(image_user_id_annotated) REFERENCES user(user_id),
  -- if the species is not null, we need to know the user that set it
  -- if the species is null, no user could have set it
  check ((image_species_id is null and image_user_id_annotated is null and image_date_annotated is null)
          or (image_species_id is not null and image_user_id_annotated is not null and image_date_annotated is not null))
);
