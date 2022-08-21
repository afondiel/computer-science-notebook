-- Course 6: SQL for data science --

/*
* Tool : DatasetteDatabase Used in this Lab
* dataset source: Film Locations in San Francisco under a PDDL: Public Domain Dedication and License.
*/

--SELECT Statement => Examples 

/*Retrieve all records with all columns from the "FilmLocations" table*/
--SELECT * FROM FilmLocations

/*Retrieve the names of all films with director names and writer names.*/
--SELECT Title, Director, Writer FROM FilmLocations;

/*
* Retrieve the names of all films released 
* in the 21st century and onwards (release years after 2001 including 2001), 
* along with filming locations and release years.
*/

--SELECT Title, ReleaseYear, Locations FROM FilmLocations;
	--WHERE ReleaseYear >= '2001'

-- Task C: Practice exercises on SELECT statement
--SELECT Locations, FunFacts FROM FilmLocations;
--SELECT Title, ReleaseYear, Locations FROM FilmLocations
--WHERE ReleaseYear<=2000;
--SELECT Title, ProductionCompany, Locations, ReleaseYear FROM FilmLocations 
-- WHERE Writer<>"James Cameron";

-- # COUNT, DISTINCT, LIMIT => Examples

-- ### Retrieve the number of rows from the "FilmLocations" table ###
-- SELECT COUNT(*) FROM FilmLocations;
/*
Retrieve the number of locations of the films which are written by James Cameron.
*/

-- SELECT COUNT(Locations) 
-- FROM FilmLocations 
-- WHERE Writer="James Cameron";


/*Retrieve the number of locations of the films which are directed by Woody Allen*/
-- SELECT COUNT(Locations) FROM FilmLocations WHERE Director="Woody Allen";


/*Retrieve the number of films shot at Russian Hill.*/

--SELECT Count(Title) FROM FilmLocations WHERE Locations="Russian Hill";

/*Retrieve the number of rows having a release year older than 1950 from the "FilmLocations" table*/

-- SELECT Count(*) FROM FilmLocations WHERE ReleaseYear<1950;


-- DISTINCT
/*Retrieve the name of all films without any repeated titles.*/
-- SELECT DISTINCT Title FROM FilmLocations;

/* Retrieve the number of release years of the films distinctly, produced by Warner Bros. Pictures .*/
-- SELECT COUNT(DISTINCT ReleaseYear) FROM FilmLocations WHERE ProductionCompany="Warner Bros. Pictures";

/*Retrieve the name of all unique films released in the 21st century and onwards, along with their release years*/
-- SELECT DISTINCT Title, ReleaseYear FROM FilmLocations WHERE ReleaseYear>=2001;

/*Retrieve the names of all the directors and their distinct films shot at City Hall*/
-- SELECT DISTINCT Title, Director FROM FilmLocations WHERE Locations="City Hall";

/*Retrieve the number of distributors distinctly who distributed films acted by Clint Eastwood as 1st actor.*/
--SELECT COUNT(DISTINCT Distributor) FROM FilmLocations WHERE Actor1="Clint Eastwood";



-- LIMIT


/*Retrieve the first 25 rows from the "FilmLocations" table.*/
-- SELECT * FROM FilmLocations LIMIT 25;

/*Retrieve the first 15 rows from the "FilmLocations" table starting from row 11*/
--SELECT * FROM FilmLocations LIMIT 15 OFFSET 10;

/*Retrieve the name of first 50 films distinctly.

*/
-- SELECT DISTINCT Title FROM FilmLocations LIMIT 50;

/*Retrieve first 10 film names distinctly released in 2015.*/
-- SELECT DISTINCT Title FROM FilmLocations WHERE ReleaseYear=2015 LIMIT 10;

/*Retrieve the next 3 film names distinctly after first 5 films released in 2015.*/

-- SELECT DISTINCT Title FROM FilmLocations WHERE ReleaseYear=2015 LIMIT 3 OFFSET 5;


/* Task A: Example exercises on INSERT */
--Ex.1 INSERT
--Insert a new instructor record with id 4 for Sandip Saha who lives in Edmonton, CA into the "Instructor" table.
-- INSERT INTO Instructor(ins_id, lastname, firstname, city, country)
-- VALUES(4, 'Saha', 'Sandip', 'Edmonton', 'CA');

-- Insert two new instructor records into the "Instructor" table. First record with id 5 for John Doe who lives in Sydney, AU. Second record with id 6 for Jane Doe who lives in Dhaka, BD.
-- INSERT INTO Instructor(ins_id, lastname, firstname, city, country)
-- VALUES(5, 'Doe', 'John', 'Sydney', 'AU'), (6, 'Doe', 'Jane', 'Dhaka', 'BD');

/*Task B: Practice exercises on INSERT*/


--Ex.2 UPDATE
/* Update the city for Sandip to Toronto*/

-- UPDATE Instructor 
-- SET city='Toronto' 
-- WHERE firstname="Sandip";

/* Update the city and country for Doe with id 5 to Dubai and AE respectively.*/
-- UPDATE Instructor 
-- SET city='Dubai', country='AE' 
-- WHERE ins_id=5;


--Ex.3 DELETE

/*Remove the instructor record of Doe whose id is 6.*/
-- DELETE FROM instructor
-- WHERE ins_id = 6;

/*Remove the instructor record of Hima.

*/

-- DELETE FROM instructor
-- WHERE firstname = 'Hima';

-- SELECT * FROM Instructor;
