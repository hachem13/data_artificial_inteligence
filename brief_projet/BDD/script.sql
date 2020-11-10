# creation de la base de données netflix

CREATE database netflix;

USE netflix;

# creation des tables netflix_shows et netflix_titles

CREATE TABLE netflix_shows(
title varchar(64)            
, rating varchar(9)          
, ratingLevel varchar(126)      
, ratingDescription int  
, release_year int
, user_rating_score int
, user_rating_size float
);

CREATE TABLE netflix_title (
show_id INT NOT NULL,
type VARCHAR (8),
title VARCHAR (105),
director VARCHAR (209),
cast VARCHAR (778),
country VARCHAR (124),
date_added DATE NOT NULL,
release_year INT NOT NULL,
rating VARCHAR (9),
duration VARCHAR (11),
listed_in VARCHAR (80),
description VARCHAR (251),
 PRIMARY KEY (show_id)
);

# implémentation des données

LOAD DATA LOCAL INFILE '/home/hachem/Documents/git/data_artificial_inteligence/brief_projet/BDD/Netflix_Shows.csv' 
INTO TABLE netflix.netflix_shows
CHARACTER SET latin1
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


LOAD DATA LOCAL INFILE '/home/hachem/Documents/git/data_artificial_inteligence/brief_projet/BDD/netflix_titles.csv' 
INTO TABLE netflix.netflix_title
CHARACTER SET latin1
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;

# Affichage de  tous les titres de films de la table netflix_titles dont 
# l’ID est inférieur strict à 80000000 

SELECT title FROM netflix.netflix_title where show_id < 80000000 ;

# Affichage de toutes les durée des TV Show (colonne duration)

SELECT duration, netflix_title.type FROM netflix.netflix_title WHERE netflix_title.type = "Tv Show";

# Affichage de tous les noms de films communs aux 2 tables (netflix_titles et netflix_shows) 

SELECT title FROM netflix.netflix_title AS nt
join netflix.netflix_shows as ns ON ns.title = nt.tittle GROUP BY title ORDER BY title;

#  Calcule de la durée totale de tous les films (Movie)S de votre table netflix_titles 

SELECT SUM(duration) AS durée_totale_de_film FROM netflix.netflix_title WHERE type = "Movie";

# Comptage du nombre de TV Shows de votre table ‘netflix_shows’ dont le ‘ratingLevel’ est renseigné

SELECT COUNT(ratingLevel) FROM netflix.netflix_shows WHERE ratingLevel <> '';

# Compter les films et TV Shows pour lesquels les noms (title) sont les mêmes sur les 
# 2 tables et dont le ‘release year’ est supérieur à 2016. 

SELECT count(netflix_title.title) AS nombre_de_titre, netflix_shows.title
from netflix_title
INNER JOIN netflix_shows ON netflix_title.title = netflix_shows.title
WHERE netflix_shows.title = netflix_title.title 
and(netflix_title.release_year > 2016 and netflix_shows.release_year > 2016) 
GROUP BY netflix_shows.title;

# Suppression de la colonne ‘rating’ de la table ‘netflix_shows’ 

ALTER TABLE netflix.netflix_shows DROP COLUMN rating;

#  Suppression des 100 dernières lignes de la table ‘netflix_shows’ 

ALTER TABLE netflix_shows ADD `ID` INT NOT NULL AUTO_INCREMENT;
DELETE FROM netflix_shows ORDER BY ID DESC LIMIT 100 ;

# ajout d'un commentaire du champs “ratingLevel” pour le TV show “Marvel's Iron Fist” de 
#la table ‘netflix_shows’ 

UPDATE netflix_shows SET ratingLevel = "it's a comment" WHERE ID = 26 AND title = "Marvel's Iron Fist";

