# Contributions - John

# Define class
class UserInterface:
    # Initialise with a pass to return immediately
    def __init__(self):
        pass

    # Define function for inputs, accepting arguments for default values
    def Inputs(self, meanLand, meanBuilding):
        # Number of bedrooms (stored as a float)
        bedrooms = float(input("How many bedrooms would you prefer?"))

        # Number of bathrooms (stored as a float)
        bathrooms = float(input("How many bathrooms would you prefer?"))

        # Number of car parking spaces (stored as a float)
        car = float(input("How many car parking spaces would you prefer?"))

        # Size of land (stored as a float). Accepts 0 if unknown
        landsize = float(input("What landsize would you prefer?"))

        # Building area (stored as a float). Accepts 0 if unknown
        buildingarea = float(input("What building area would you prefer?"))

        # Latitude (stored as a string). Accepts 0 if unknown
        latitude = input("Would you prefer 'north' or 'south'?")

        # Longitude (stored as a string). Accepts 0 if unknown
        longitude = input("Would you prefer 'east' or 'west'?")

        # Assign default value if unknown
        if landsize == 0:
            landsize = meanLand  # Average size for land
        
        # Assign default value if unknown
        if buildingarea == 0:
            buildingarea = meanBuilding  # Average size for building area

        # Convert input to usable coordinate
        if latitude == 'north':
            latitude = -37.5903
        elif latitude == 'south':
            latitude = -37.9904
        # Assign central value if unknown
        elif latitude == '0':
            latitude = -37.7903
        else:
            print("Error! Please enter a valid latitude")

        # Convert input to usable coordinate
        if longitude == 'east':
            longitude = 145.2507
        elif longitude == 'west':
            longitude = 144.6994
        # Assign central value if unknown
        elif longitude == '0':
            longitude = 144.9751
        else:
            print("Error! Please enter a valid longitude")

        # Return processed inputs for subsequent usage
        return bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude
