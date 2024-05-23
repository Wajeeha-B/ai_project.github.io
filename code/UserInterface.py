import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt
import numpy as np

class UserInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.result = None

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
        
        self.result = (bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude)

        return self.result

    def gui_inputs(self, data):
        self.setWindowTitle("User Preferences")
        self.setGeometry(100, 100, 600, 600)

        layout = QVBoxLayout()

        # Number of bedrooms
        layout.addWidget(QLabel("How many bedrooms would you prefer?"))
        self.bedrooms_slider, self.bedrooms_label = self.create_slider(1, 10)
        layout.addLayout(self.bedrooms_slider)

        # Number of bathrooms
        layout.addWidget(QLabel("How many bathrooms would you prefer?"))
        self.bathrooms_slider, self.bathrooms_label = self.create_slider(int(data['Bathroom'].min()), int(data['Bathroom'].max()))
        layout.addLayout(self.bathrooms_slider)

        # Number of car parking spaces
        layout.addWidget(QLabel("How many car parking spaces would you prefer?"))
        self.car_slider, self.car_label = self.create_slider(int(data['Car'].min()), int(data['Car'].max()))
        layout.addLayout(self.car_slider)

        # Size of land
        layout.addWidget(QLabel("What landsize would you prefer?"))
        self.landsize_slider, self.landsize_label = self.create_decimal_slider(data['Landsize'].min(), data['Landsize'].max())
        layout.addLayout(self.landsize_slider)

        # Building area
        layout.addWidget(QLabel("What building area would you prefer?"))
        self.buildingarea_slider, self.buildingarea_label = self.create_decimal_slider(data['BuildingArea'].min(), data['BuildingArea'].max())
        layout.addLayout(self.buildingarea_slider)

        # Latitude
        layout.addWidget(QLabel("What latitude would you prefer?"))
        self.latitude_slider, self.latitude_label = self.create_decimal_slider(data['Latitude'].min(), data['Latitude'].max())
        layout.addLayout(self.latitude_slider)

        # Longitude
        layout.addWidget(QLabel("What longitude would you prefer?"))
        self.longitude_slider, self.longitude_label = self.create_decimal_slider(data['Longitude'].min(), data['Longitude'].max())
        layout.addLayout(self.longitude_slider)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.gui_submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def create_slider(self, min_value, max_value):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        label = QLabel(f"{min_value}")
        
        def update_label(value):
            label.setText(f"{value}")
        
        slider.valueChanged.connect(lambda value: update_label(value))
        
        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(label)
        
        return layout, label

    def create_decimal_slider(self, min_value, max_value, num_steps=1000):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(num_steps)
        label = QLabel(f"{min_value:.2f}")
        
        def update_label(value):
            scaled_value = min_value + (max_value - min_value) * (value / num_steps)
            label.setText(f"{scaled_value:.2f}")
        
        slider.valueChanged.connect(lambda value: update_label(value))
        
        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(label)
        
        return layout, label

    def gui_submit(self):
        bedrooms = int(self.bedrooms_label.text())
        bathrooms = int(self.bathrooms_label.text())
        car = int(self.car_label.text())
        landsize = float(self.landsize_label.text())
        buildingarea = float(self.buildingarea_label.text())
        latitude = float(self.latitude_label.text())
        longitude = float(self.longitude_label.text())

        self.result = (bedrooms, bathrooms, car, landsize, buildingarea, latitude, longitude)
        self.close()
        QApplication.instance().quit()  # Ensure the event loop terminates

    def get_result(self):
        return self.result

    def get_houses_within_range(self, data, pred, latitude, longitude, buildingarea, bathrooms, bedrooms, car, tolerance=0.05):
        tolerance_price = pred * tolerance
        tolerance_latitude = latitude * tolerance * 0.1
        tolerance_longitude = longitude * tolerance * 0.1
        tolerance_buildingarea = buildingarea * tolerance
        tolerance_bathrooms = 0
        tolerance_bedrooms = 0
        tolerance_car = 0

        houses = data[
            (np.abs(data['Latitude'] - latitude) <= np.abs(tolerance_latitude)) &
            (np.abs(data['Longitude'] - longitude) <= np.abs(tolerance_longitude)) &
            (np.abs(data['BuildingArea'] - buildingarea) <= np.abs(tolerance_buildingarea)) &
            (np.abs(data['Bathroom'] - bathrooms) <= np.abs(tolerance_bathrooms)) &
            (np.abs(data['Bedroom'] - bedrooms) <= np.abs(tolerance_bedrooms)) &
            (np.abs(data['Car'] - car) <= np.abs(tolerance_car))
        ]
        return houses
