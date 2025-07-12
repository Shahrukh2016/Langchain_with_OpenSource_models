from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name' : 'Shahrukh', 'age' : 27}

print(new_person)