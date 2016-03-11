drop table if exists user;
drop table if exists image;
drop table if exists species;
pragma foreign_keys = ON;
create table user (
  user_id integer primary key autoincrement,
  user_username text not null unique,
  user_password text not null
);
create table species (
  species_id integer primary key autoincrement,
  species_name text not null unique
);
create table image (
  image_id integer primary key autoincrement,
  image_filepath text not null unique,
  image_height integer not null,
  image_width integer not null,
  image_species_id integer,
  image_user_id integer,
  FOREIGN KEY(image_species_id) REFERENCES species(species_id),
  FOREIGN KEY(image_user_id) REFERENCES user(user_id)
);
