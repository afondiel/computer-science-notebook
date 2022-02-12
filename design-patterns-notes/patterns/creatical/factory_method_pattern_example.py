"""! @brief Factory Design Pattern Exampe """
#
# @author : afondiel | 12/02/2022
#

from abc import ABCMeta, abstractmethod, abstractstaticmethod
from enum import IntEnum

class PersonEnum(IntEnum):
    STUDENT = 0
    TEACHER = 1
    # UNKNOWN = 5
    
# Abstract Creator       
class IPersonFactory(metaclass=ABCMeta):
    "The PersonFactory interface indicating all the methods to implement"
    @abstractstaticmethod
    def person_method():
        "A method to implement"

# Concrete Creator       
class PersonFactory(IPersonFactory):
    "The PersonFactory Concrete Class"
    def __init__(self):
        self.info = 'Factory Creator'
    
    # Parameterized factory methods    
    def CreatePerson(person_type):
        if person_type == PersonEnum.STUDENT:
             return Student(); 
        if person_type == PersonEnum.TEACHER:
             return Teacher();
        else : 
            return print('unkown person type')
            
# Products                     
class Student(IPersonFactory):
    def __init__(self):
        self.name = "Student Object"
        
    def person_method(self):
        print(self.name)
        
class Teacher(IPersonFactory):
    def __init__(self):
        self.name = "Teacher Object"
        
    def person_method(self):
        print(self.name)
        
# CLIENT VIEW : MAiN TEST
# creation of student and teacher instances(objects) using FACTORY 
# 
if __name__ == '__main__':

    # create/manufactorying student object  
    student = PersonFactory.CreatePerson(PersonEnum.STUDENT)
    # create/manufactorying teacher object  
    teacher = PersonFactory.CreatePerson(PersonEnum.TEACHER)
    
    # objects usage
    student.person_method()
    teacher.person_method()

