class Parent:
    def __init__(self):
        print("Parent __init__")

class Child(Parent):
    def __init__(self):
        super().__init__()  # 调用父类的__init__方法
        print("Child __init__")

child = Child()
