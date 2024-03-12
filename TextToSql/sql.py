import sqlite3

## Connect to sqlite

connection = sqlite3.connect("student.db")

## Create a cursor object to insert record, create table, retrieve
cursor = connection.cursor()

## create table
table_info="""
Create table STUDENTs(NAME VARCHAR(25), CLASS VARCHAR(25),
SECTION VARCHAR(25), MARKS INT);
"""

cursor.execute(table_info)

## Insert some more records

cursor.execute('''Insert Into STUDENTS values('Krish', 'Data Science', 'A', '90')''')
cursor.execute('''Insert Into STUDENTS values('Sudhanshu', 'Data Science', 'B', '100')''')
cursor.execute('''Insert Into STUDENTS values('Darius', 'Data Science', 'A', '86')''')
cursor.execute('''Insert Into STUDENTS values('Vikas', 'DevOps', 'A', '50')''')
cursor.execute('''Insert Into STUDENTS values('Dipesh', 'DevOps', 'A', '35')''')

## Display all the records
print("The inserted records are")

data=cursor.execute('''Select * From STUDENTS''')

for row in data:
    print(row)

## Close the connection

connection.commit()
connection.close()
