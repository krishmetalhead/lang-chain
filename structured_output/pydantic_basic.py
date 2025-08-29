from pydantic import BaseModel , EmailStr , Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    income: int = Field(ge=500)

new_student = {'name':'krishnendu' , 'income': 500 , 'email':'krish.b@gmail.com'}

student_dict = dict(Student(**new_student))

print(student_dict['name'])
print(student_dict['income'])
print(student_dict['email'])
print(student_dict['age'])

student_json = Student(**new_student).model_dump_json()

print(student_json)
