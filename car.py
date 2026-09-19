

class Car:
    def __init__(self,make,model,year,color):
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.odometer_reading = 0
    def get_descriptive_name(self):
        long_name = f'{self.year} {self.make} {self.model}'
        return long_name.title()

    def rear_odometer(self):
        print(f'this car has {self.odometer_reading} miles on it')

    def update_odometer(self,mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage         
        else:
            print("you can't roll back an odometer")

        print(f'This car can go about {range} miles on a full charge')
        def increments_odometer(self,miles):
            self.odometerreading += miles
 
class Battrey:
    def __init__(self,battrey_size=40):
        self.battrey_size = battrey_size

    def describe_battrey(self):
        print(f'this car has a {self.battrey_size}-KWH battrey.')    

    def get_range(self):
        if self.battrey_size == 40:
            range = 150
        elif self.battrey_size == 65:
            range = 225

        print(f'This car can go about {range} miles on a full charge')  

    def upgrade_battrey(self):
        if self.battrey_size == 65:
            range = 225
        else:
            range = 225                     

        print(f'This car can go about {range} miles on a full charge')

class Electriccar(Car):

    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battrey = Battrey()

    def describe_battrey(self):
        print(f'this car has a {self.battrey_size}-KWH battrey')    




        