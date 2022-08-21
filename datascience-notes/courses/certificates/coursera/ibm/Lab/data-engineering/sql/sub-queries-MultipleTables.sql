-- Sub-queries : 
-- 1 : Retrieve only the EMPLOYEES records that correspond to jobs in the JOBS table.
select * from employees 
where JOB_ID IN (select JOB_IDENT from jobs);

-- 2 :Retrieve only the list of employees whose JOB_TITLE is Jr. Designer.
select * from employees 
where JOB_ID IN (select JOB_IDENT from jobs where JOB_TITLE= 'Jr. Designer'); 


-- 3 : Retrieve JOB information and who earn more than $70,000.
select JOB_TITLE, MIN_SALARY,MAX_SALARY,JOB_IDENT from jobs 
where JOB_IDENT IN (select JOB_ID from employees where SALARY > 70000 );

-- 4 : Retrieve JOB information and whose birth year is after 1976.
select JOB_TITLE, MIN_SALARY,MAX_SALARY,JOB_IDENT from jobs 
where JOB_IDENT IN (select JOB_ID from employees where YEAR(B_DATE)>1976 );

-- 5  : Retrieve JOB information for female employees whose birth year is after 1976.
select JOB_TITLE, MIN_SALARY,MAX_SALARY,JOB_IDENT from jobs 
where JOB_IDENT IN (select JOB_ID from employees where YEAR(B_DATE)>1976 and SEX='F' );

-- Implicit JOIN 
-- 1 : 
select * from employees, jobs;
-- 2 : 
select * from employees, jobs 
where employees.JOB_ID = jobs.JOB_IDENT;
-- 3 : 
select * from employees E, jobs J 
where E.JOB_ID = J.JOB_IDENT;
-- 4 : 
select EMP_ID,F_NAME,L_NAME, JOB_TITLE from employees E, jobs J
 where E.JOB_ID = J.JOB_IDENT;
-- 5 :
select E.EMP_ID,E.F_NAME,E.L_NAME, J.JOB_TITLE from employees E, jobs J where E.JOB_ID = J.JOB_IDENT;

