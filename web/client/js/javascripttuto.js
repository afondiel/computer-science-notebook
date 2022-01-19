/* C 'est le moment d'apprendre du JAVASCRIPT 'CAUSE IS FUNNNN :) */


/*---------------------- Variables --------------------------*/
/*var number = 2;
alert(typeof number); // Affiche : « number »
var text = 'Mon texte';
alert(typeof text); // Affiche : « string »
var aBoolean = false;
alert(typeof aBoolean); // Affiche : « boolean »
Simple non ? Et maintenant voici comment tester l'existence d'une variable :
alert(typeof nothing); // Affiche : « undefined » Voilà*/

////////////////////////////////////////////////////////////////
//var num1 = 1;
//var num2 = 2;
//alert(num1+num2);

/*-------------------Conditions --------------------------*/
/*var test = true;

var age = prompt('Quel votre age?');

if(age<18){
	alert ('Vous êtes minuer!' );
}
else {alert('Vous êtes majeur');}
*/

/*------------------------------------FUNCTIONS------------------------------------*/
//

/*function add(num1,num2){
	return num1+num2;
}
alert(add(1,2));*/

/*function yourID(name, age){
	if(age<18){
		alert(name);
		alert ('You are under 18 !' );
	}
	else {
		alert(name);
		alert('You are above 18 ');
	}
}
yourID('Afonso DIELA', 27);*/

/*--------------------- ARRAY & LOOPS ----------------------*/
/*var myList    = ['focus', 'success','light','ambition'];
var myNewList = [12, yourID];
myNewList[2]  = myList;

myList[0] = 'nao'; 					//changing an element 
myList[4] = 'me'; 					//adding a new element 

myList.forEach(yourID('Afonso DIELA', 27)){
	alert('Afonso DIELA');
}
*/
/*push() : ajoute un ou plusieurs éléments à la fin du tableau (un argument par élément ajouté) et retourne la nouvelle
taille de ce dernier.
pop() : retire et retourne le dernier élément d'un tableau.
unshift() : ajoute un ou plusieurs éléments au début du tableau (un argument par élément ajouté) et retourne la
nouvelle taille de ce dernier.
shift() : retire et retourne le premier élément d'un tableau.*/

//////////////LOOPS/////////////////
/*var test = true;
while(test){
	alert('This is true');
	test = false;
}*/
/*???infinity loop???*/
//var nb1 = 4, nb2 = 5;
//while (nb1 < nb2) {
// To avoid
//}

/*??? Dangerous Loop ???
var times=0;
do {
	//Instructions
	console.log('loged',times);
}while(times<5);*/

/*for (var i = 0 ; i < 4 ; i++) {
	console.log('i is',i);
}*/
/*var myList    = ['focus', 'success','light','ambition'];
for (var i = 0 ; i < myList.length; i++) {
	console.log('my list '+myList+ ' of elements');
}*/
/*forEach (function) {								//NOT supported by IE8 
	//Instructions
}

for in(){
		//Instructions

}*/

/*every (stops looping the first time the iterator returns false or something falsey)
some (stops looping the first time the iterator returns true or something truthy)
filter (creates a new array including elements where the filter function returns true and omitting the ones where it returns false)
map (creates a new array from the values returned by the iterator function)
reduce (builds up a value by repeated calling the iterator, passing in previous values; see the spec for the details; useful for summing the contents of an array and many other things)
reduceRight (like reduce, but works in descending rather than ascending order)*/

////////////////////////////////////OBJECTS/////////////////////////////////////////////

/*
* var myString = 'Ceci est une chaîne de caractères'; // On crée un objet String
* alert(myString.length); 			                  // On affiche le nombre de caractères, au moyen de la propriété « length »
* alert(myString.toUpperCase()); 					  // On récupère la chaîne en majuscules, avec la méthode toUpperCase()*/

/*
* var myString = 'Test';
* alert(myString.length); 		//Affiche : « 4 »
* myString = 'Test 2';
* alert(myString.length); 		//Affiche : « 6 » (l'espace est aussi un caractère)*/

/*----------------------------------- Selector methods --------------------------------------*/


//Selector methods are:
/*document.getElementsByTagName('div')
document.getElementsByClassName('done')
document.getElementById('my-id')
document.querySelector('#my-id')
document.querySelectorAll('.classname')

Once you have selected an html element, you can modify it:
document.getElementById('my-id').innerHTML = "new html"
document.getElementById('my-id').className = "newclass otherclass"*/

/*-------+---------+---------+--------+--- EVENTS ----+-------+-------+------+-------+-------*/


/*Popular Javascript Events Are:
- click
- mouseenter
- mouseleave
- mousedown
- mouseup
- mousemove
- keydown
- keyup
- blur
- focus*/

/*--------------------------------------- APIs/DOM --------------------------------------------*/













/*-------------------------------------------------------------------------------------*/
