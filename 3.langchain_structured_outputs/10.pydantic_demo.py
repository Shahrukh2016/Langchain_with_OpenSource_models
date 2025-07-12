# from pydantic import BaseModel

# class Student(BaseModel):
#     name : str

# new_student = {'name' : 'Shahrukh'}
# student = Student(**new_student)
# print(new_student)

#####################################################################################################

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = 'Elon'
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt = 0, lt = 10, default = 5, description = 'A decimal value representing the cgpa of a student')
    
new_student = {'name' : 'srk', 'age' : 22, 'email' : 'abc@gmail.com', 'cgpa' : 9}
student = Student(**new_student)

print(student.model_dump)
print(student.model_dump_json)