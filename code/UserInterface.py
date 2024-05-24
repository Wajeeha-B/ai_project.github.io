import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QHBoxLayout, QScrollArea
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
        avg_bedrooms = int(data['Bedroom'].mean())
        self.bedrooms_slider, self.bedrooms_label = self.create_slider(int(data['Bedroom'].min()), int(data['Bedroom'].max()), avg_bedrooms)
        layout.addLayout(self.bedrooms_slider)

        # Number of bathrooms
        layout.addWidget(QLabel("How many bathrooms would you prefer?"))
        avg_bathrooms = int(data['Bathroom'].mean())
        self.bathrooms_slider, self.bathrooms_label = self.create_slider(int(data['Bathroom'].min()), int(data['Bathroom'].max()), avg_bathrooms)
        layout.addLayout(self.bathrooms_slider)

        # Number of car parking spaces
        layout.addWidget(QLabel("How many car parking spaces would you prefer?"))
        avg_car = int(data['Car'].mean())
        self.car_slider, self.car_label = self.create_slider(int(data['Car'].min()), int(data['Car'].max()), avg_car)
        layout.addLayout(self.car_slider)

        # Size of land
        layout.addWidget(QLabel("What landsize would you prefer?"))
        avg_landsize = data['Landsize'].mean()
        self.landsize_slider, self.landsize_label = self.create_decimal_slider(data['Landsize'].min(), data['Landsize'].max(), avg_landsize)
        layout.addLayout(self.landsize_slider)

        # Building area
        layout.addWidget(QLabel("What building area would you prefer?"))
        avg_buildingarea = data['BuildingArea'].mean()
        self.buildingarea_slider, self.buildingarea_label = self.create_decimal_slider(data['BuildingArea'].min(), data['BuildingArea'].max(), avg_buildingarea)
        layout.addLayout(self.buildingarea_slider)

        # Latitude
        layout.addWidget(QLabel("What latitude would you prefer?"))
        avg_latitude = data['Latitude'].mean()
        self.latitude_slider, self.latitude_label = self.create_decimal_slider(data['Latitude'].min(), data['Latitude'].max(), avg_latitude)
        layout.addLayout(self.latitude_slider)

        # Longitude
        layout.addWidget(QLabel("What longitude would you prefer?"))
        avg_longitude = data['Longitude'].mean()
        self.longitude_slider, self.longitude_label = self.create_decimal_slider(data['Longitude'].min(), data['Longitude'].max(), avg_longitude)
        layout.addLayout(self.longitude_slider)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.gui_submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)
        self.show()

    def create_slider(self, min_value, max_value, avg_value):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        label = QLabel(f"{min_value}")

        if avg_value:
            slider.setValue(avg_value)
            label.setText(f"{avg_value}")
        
        def update_label(value):
            label.setText(f"{value}")
        
        slider.valueChanged.connect(lambda value: update_label(value))
        
        layout = QHBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(label)
        
        return layout, label

    def create_decimal_slider(self, min_value, max_value, avg_value, num_steps=1000):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(num_steps)
        label = QLabel(f"{min_value:.2f}")

        if avg_value:
            slider.setValue(int((avg_value - min_value) / (max_value - min_value) * num_steps))
            label.setText(f"{avg_value:.2f}")
        
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

    def get_closest_houses(self, data, pred, latitude, longitude, buildingarea, bathrooms, bedrooms, car, n=5, show_window=False):
        # Calculate the absolute difference for each feature
        data['Price_diff'] = np.abs(data['Price'] - pred) / (data['Price'].max() - data['Price'].min())
        data['Latitude_diff'] = np.abs(data['Latitude'] - latitude) / (data['Latitude'].max() - data['Latitude'].min())
        data['Longitude_diff'] = np.abs(data['Longitude'] - longitude) / (data['Longitude'].max() - data['Longitude'].min())
        data['BuildingArea_diff'] = np.abs(data['BuildingArea'] - buildingarea) / (data['BuildingArea'].max() - data['BuildingArea'].min())
        data['Bathrooms_diff'] = np.abs(data['Bathroom'] - bathrooms) / (data['Bathroom'].max() - data['Bathroom'].min())
        data['Bedrooms_diff'] = np.abs(data['Bedroom'] - bedrooms) / (data['Bedroom'].max() - data['Bedroom'].min())
        data['Car_diff'] = np.abs(data['Car'] - car) / (data['Car'].max() - data['Car'].min())
        
        # Calculate a total difference score (weighted sum or simple sum)
        data['Total_diff'] = (
            data['Price_diff'] +
            data['Latitude_diff'] +
            data['Longitude_diff'] +
            data['BuildingArea_diff'] +
            data['Bathrooms_diff'] +
            data['Bedrooms_diff'] +
            data['Car_diff']
        )
        
        # Sort by the total difference and get the top n houses
        closest_houses = data.sort_values(by='Total_diff').head(n)
        
        # Drop the temporary columns used for calculation
        closest_houses = closest_houses.drop(columns=['Price_diff', 'Latitude_diff', 'Longitude_diff', 'BuildingArea_diff', 'Bathrooms_diff', 'Bedrooms_diff', 'Car_diff', 'Total_diff'])
        
        if show_window:
            self.display_closest_houses(closest_houses, pred)
        
        return closest_houses

    def display_closest_houses(self, closest_houses, pred):
        display_window = QWidget()
        display_window.setWindowTitle("Closest Houses")
        display_window.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        pred_label = QLabel(f"Predicted Price: {pred:,} AUD")
        pred_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #ff6347; margin-bottom: 10px;")
        layout.addWidget(pred_label)

        title_label = QLabel("Closest Houses:")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin: 20px 0;")
        layout.addWidget(title_label)

        for i, house in closest_houses.iterrows():
            house_frame = QVBoxLayout()
            house_frame.setContentsMargins(10, 10, 10, 10)
            house_frame.setSpacing(5)
            
            price_label = QLabel(f"Price: {house['Price']:,} AUD")
            price_label.setStyleSheet("font-size: 14pt; color: #2e8b57;")
            house_frame.addWidget(price_label)

            location_label = QLabel(f"Location: Latitude {house['Latitude']}, Longitude {house['Longitude']}")
            location_label.setStyleSheet("font-size: 12pt; color: #2e8b57;")
            house_frame.addWidget(location_label)

            details_label = QLabel(f"Building Area: {house['BuildingArea']} sqm, Bathrooms: {house['Bathroom']}, Bedrooms: {house['Bedroom']}, Car Spaces: {house['Car']}")
            details_label.setStyleSheet("font-size: 12pt; color: #2e8b57;")
            house_frame.addWidget(details_label)
            
            house_container = QWidget()
            house_container.setLayout(house_frame)
            house_container.setStyleSheet("""
                QWidget {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    margin: 10px 0;
                    padding: 10px;
                }
            """)
            layout.addWidget(house_container)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(layout)
        scroll_area.setWidget(scroll_content)

        main_layout = QVBoxLayout(display_window)
        main_layout.addWidget(scroll_area)

        display_window.setLayout(main_layout)
        display_window.show()

        # Start a new event loop for the display window
        display_app = QApplication.instance() or QApplication(sys.argv)
        display_app.exec_()
