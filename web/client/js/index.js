// TEST MOSH

//Global variables

var chaine = "f#ck the society"; 
display_content(chaine);

var number = 2;
//reuseble functions

function double(a){
    return a*2;
}

function display_content(input){
    console.log(input);    
}



display_content(double(number));