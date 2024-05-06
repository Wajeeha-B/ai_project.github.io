# Contributions - John
class UserInterface:
    def __init__(self):
        pass

    def Inputs(self,meanLand,meanBuilding):
        # Dwelling type
        type = input("What type of dwelling would you prefer? Either 'u' for unit or 'h' for house")
        # Number of bedrooms
        bedrooms = input("How many bedrooms would you prefer?")
        # Number of bathrooms
        bathrooms = input("How many bathrooms would you prefer?")
        # Number of car parking spaces
        car = input("How many car parking spaces would you prefer?")
        # Size of land
        size = input("What size would you prefer?")
        # Latitude
        latitude = input("Would you prefer 'north' or 'south'?")
        # Longitude
        longitude = input("Would you prefer 'east' or 'west'?")

        if size == 0:
            if type == 'u':
                size = meanBuilding # Average size for unit
            if type == 'h':
                size = meanLand # Average size for house

        if latitude == 'north':
            latitude = -37.5903
        elif latitude == 'south':
            latitude = -37.9904
        elif latitude == 0:
            latitude = -37.7903
        elif latitude != 'north' & latitude != 'south' & latitude != 0:
            print("Error! Please enter a valid latitude")

        if longitude == 'east':
            longitude = 145.2507
        elif longitude == 'west':
            longitude == 144.6994
        elif longitude == 0:
            longitude = 144.9751
        elif longitude!= 'east' & longitude != 'west' & longitude != 0:
            print("Error! Please enter a valid longitude")
        
        return type, bedrooms, bathrooms, car, size, latitude, longitude