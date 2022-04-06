#-------------------------- mycode-------------------------------------#

#$var1=2;
#$var2=4;
#-------------------------------------c coding -----------------------------------------------#
# $var1 = $var2 - $var1;
# $var2 = $var2 + $var1;
# $var2 = $var2 + $var1;

# print $var1,"\n", $var2, "\n";


#--------------------------------------------------------------------------------------------#

#EX: 1 ($var2 ,$var1 )=($var1 ,$var2 );
#print $var1,"\n", $var2, "\n";


#-------------------- List--------------------------------------------
#@house =('bed','chair','table');
#$#house : to print the last element of the list
# @house[1..4]=('elt', 'e' , '3' ,'et');
# print @house;

# ------------------------------ Hash---------------------------------
#%hash=(   "elt1" => ['a','b', 'c'],
		  # "elt2" => [[1,2],['a','b', 4]]);
# print "\n";
# print %hash;
# print "\n";
# %longday=(
	# 'Sun' =>"\nSunday",
	# 'Mon' =>"Monday",
	# 'Tue' =>"Tuesday"
# );
# print $longday{'Sun'},"\n";

#----------------------------------files&handles------------------------
# print "Enter a number: ";   #Ask for a number
# #$number = <STDIN>; 	    #Get the number 
# #chop($number = <STDIN>); 
# chomp($number = <STDIN>); 
# print $number; 				#Display the number 
#print "toto"x3, "\n";
# $lgconsole = 80;
# $string = "keyboardER !\n";
# print " " x (($lgconsole - length string)/2), $string;
# open (FILE, "file") or die "Can't open notes files : $ !\n";
# while($line = <FILE>){
	# if ($line =~ /([^:]+)larry:/){
		# print $1;
	# }
# }


#------------------------ regular expressions-----------------------------
# print /\d{7;11}+/;
# $line = "totolarry:@98# larry:toto\9 @ x bdvbv 97-'1 larry:";
	# if ($line =~ /([^:]+)larry:/){
		# print $1;
	# }
# $s = "fred xxxxxxxxxxxxxxxx toto";
# $s=~ s/x+// ; 
# print $s;
# $line1 = "toto 64.45 toto";
	# if ($line =~ /toto/){
		# print $1;
 # }  
 #-----------------1st method-------------------------------
#$line1 = "Aujourd'hui il y a du du du du du du soleil \n";
# while ($line1 =~ /\b(\w+)\s+\1/){
	# $line1 =~ s/\b(\w+)\s+\1/$1/g;	
# }
# print $line1;
 #-------------------2nd method--------------------------------
#$line1 =~ s/\b(\w+)(\s+\1)+/$1/g;
#print $line1;
#----------------------REGEX : Regular Expression-----------------------------
# $string = "password=xyzz verbose=9 score=0" ;
# %hash = $string =~ /(\w+)=(\w+)/g;
# print $hash;
#--------------------------------------------------------------------------

#-------------------- Functions-----------------------------------------------

# sub somme{
	# #var1 and var2 are the arguments of the function 
	# my ($var1, $var2)=@_; 												#@ : passage par reference
	# my $res = $var1 + $var2;
	# return $res;
# }

# print somme(2,3);

# sub menu{  #like void() in c programming	

   # $teame1=41;
   # $teame2=100;
   # #$score = $time1"x"$time2; 
   # print "======================== Game of the day ==============================\n";
   # print "===========================Football =====================================\n";
   # print "======================== PSG x Real Madrid ====== $teame1 x $teame2 ===\n";
   # print "======================== FCB x Marseille   =============================\n\n";
   # print "=========================== Basketball ====================================\n";
   # print "======================== LA Laskers x GSW ==============================\n";
   # print "======================== Chicago x Cavaliers ===========================\n";

# }

# &menu;

#--------------------REGEX-------------------------------------
  #metacharacter(s) ;; the metacharacters column specifies the regex syntax being demonstrated
  #=~ m//          ;; indicates a regex match operation in Perl
  #=~ s///         ;; indicates a regex substitution operation in Perl
  





#-------------------------------------- end -------------------------------------------------#