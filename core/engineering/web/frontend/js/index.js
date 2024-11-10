// TEST MOSH

//00:00 Qu'est-ce que JavaScript
// 04:41 Configuration de l'environnement de développement 
// 07:52 JavaScript dans les navigateurs
// 11:41 Séparation des préoccupations
// 13:47 JavaScript dans Node
// 16:11 Variables
// 21:49 Constantes
// 23:35 Types primitifs
// 26:47 Saisie dynamique
// 30:06 Objets
// 35:22 Tableaux
// 39:41 Fonctions
// 44:22 Types de fonctions

//00:00 Qu'est-ce que JavaScript
// 04:41 Configuration de l'environnement de développement 
// 07:52 JavaScript dans les navigateurs
// 11:41 Séparation des préoccupations
// 13:47 JavaScript dans Node
// 16:11 Variables
// 21:49 Constantes
// 23:35 Types primitifs
// 26:47 Saisie dynamique
// 30:06 Objets
let name, age;

let person = {
    name : 'robot',
    age : 30
};

//console.log(person);
// console.log(person.name = 'newname');
//console.log(person['name'] = 'namenew');

// 35:22 Tableaux
let arrayColors = ['red', 'blue'];
arrayColors[2] = 'black';
//Dynamic change
arrayColors[2] = 0;
console.log(arrayColors);
//Object approach
console.log(arrayColors.length);

// 39:41 Fonctions

//void function
function greet(){
    console.log('HELLO');
}

//Parameters
function greet(name){
    console.log('HELLO' + name);
}

//Call
greet();
//in the call function receives an argument
greet(' aD');

// 44:22 Types de fonctions


//Global variables
var chaine = "f#ck the society"; 
display_content(chaine);

var number = 2;
var bool = false;
//alert(typeof bool);

//reusable functions

function double(a){
    return a*2;
}

function display_content(input){
    console.log(input);    
}


// MAIN

// display_content(double(number));
// display_content(typeof bool);