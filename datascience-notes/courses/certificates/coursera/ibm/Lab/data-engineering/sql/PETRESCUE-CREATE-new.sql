-- Drop the PETRESCUE table in case it exists
--drop table PETRESCUE;
-- Create the PETRESCUE table 
--create table PETRESCUE (
	--ID INTEGER NOT NULL,
	--ANIMAL VARCHAR(20),
	--QUANTITY INTEGER,
	--COST DECIMAL(6,2),
	--RESCUEDATE DATE,
	--PRIMARY KEY (ID)
	----);
-- Insert sample data into PETRESCUE table
--insert into PETRESCUE values 
	--(1,'Cat',9,450.09,'2018-05-29'),
	--(2,'Dog',3,666.66,'2018-06-01'),
	--(3,'Dog',1,100.00,'2018-06-04'),
	--(4,'Parrot',2,50.00,'2018-06-04'),
	--(5,'Dog',1,75.75,'2018-06-10'),
	--(6,'Hamster',6,60.60,'2018-06-11'),
	--(7,'Cat',1,44.44,'2018-06-11'),
	--(8,'Goldfish',24,48.48,'2018-06-14'),
--	(9,'Dog',2,222.22,'2018-06-15')
	
--;

-- ex2 : Aggregate Functions
-- A1
select SUM(COST) from PETRESCUE;
--A2
select SUM(COST) AS SUM_OF_COST from PETRESCUE;
--A3
select MAX(QUANTITY) from PETRESCUE;
-- A4
select AVG(COST) from PETRESCUE;
-- A5
select AVG(COST/QUANTITY) from PETRESCUE where ANIMAL = 'Dog';

-- ex3 : Scalar and String Functions

--B1 :
select ROUND(COST) from PETRESCUE;
-- B2 : 
select LENGTH(ANIMAL) AS PET_LENGHT from PETRESCUE;
-- B3
select UCASE(ANIMAL) from PETRESCUE;
-- B4
select DISTINCT(UCASE(ANIMAL)) from PETRESCUE;
-- B5
select * from PETRESCUE where LCASE(ANIMAL) = 'cat';

-- ex4 : Date and Time Functions
-- C1 : day of the month
select DAY(RESCUEDATE) from PETRESCUE where ANIMAL = 'Cat';
-- C2 :  number of rescue in may
select SUM(QUANTITY) from PETRESCUE where MONTH(RESCUEDATE)='05';
-- C3 : nb of rescues in day 14
select SUM(QUANTITY) from PETRESCUE where DAY(RESCUEDATE)='14';
-- C4 : 
select (RESCUEDATE + 3 DAYS) from PETRESCUE;
--C5 : 
select (CURRENT DATE - RESCUEDATE) from PETRESCUE;






