# Contributions - John
class UserInterface:
    def __init__(self):
        pass

    def Inputs(self, meanLand, meanBuilding):
        # Number of bedrooms
        bedrooms = float(input("How many bedrooms would you prefer?"))
        # Number of bathrooms
        bathrooms = float(input("How many bathrooms would you prefer?"))
        # Number of car parking spaces
        car = float(input("How many car parking spaces would you prefer?"))
        # Size of land
        landsize = float(input("What landsize would you prefer?"))
        # Building area
        buildingarea = float(input("What building area would you prefer?"))
        # Latitude
        latitude = input("Would you prefer 'north' or 'south'?")
        # Longitude
        longitude = input("Would you prefer 'east' or 'west'?")

        if landsize == 0:
            landsize = meanLand  # Average size for land
        if buildingarea == 0:
            buildingarea = meanBuilding  # Average size for building area

        if latitude == 'north':
            latitude = -37.5903
        elif latitude == 'south':
            latitude = -37.9904
        elif latitude == '0':
            latitude = -37.7903
        else:
            print("Error! Please enter a valid latitude")

        if longitude == 'east':
            longitude = 145.2507
        elif longitude == 'west':
            longitude = 144.6994
        elif longitude == '0':
            longitude = 144.9751
        else:
            print("Error! Please enter a valid longitude")

        return bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude
