#classes
class Plant:
    def __init__(self,name,hum,temp):
        self.name=name
        self.hum=hum
        self.temp=temp
    def getName(self):
        return self.name
    def getH(self):
        return self.hum
    def getT(self):
        return self.temp
    def Display(self):
        print(self.name,"H:",self.hum,"T",self.temp)

class Collection:
    def __init__(self):
        self.garden=[]

    def getAll(self):
        return self.garden

    def getOne(self, _name):
        for i in self.garden:
            if i.getName==_name:
                return i

    def addPlant(self, _plant):
        self.garden.append(_plant)

    def DisplayAll(self):
        for i in self.garden:
            i.Display()

   #hardware:https://docs.circuitpython.org/projects/charlcd/en/latest/
import board
import digitalio
import adafruit_character_lcd.character_lcd as character_lcd


# setting up the data pins
# // initialize the library by associating any needed LCD interface pin
# // with the arduino pin number it is connected to
# const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
# LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

lcd_rs = digitalio.DigitalInOut(board.D12)
lcd_en = digitalio.DigitalInOut(board.D11)
lcd_d4 = digitalio.DigitalInOut(board.D5)
lcd_d5 = digitalio.DigitalInOut(board.D4)
lcd_d6 = digitalio.DigitalInOut(board.D3)
lcd_d7 = digitalio.DigitalInOut(board.D2)
lcd_backlight=digitalio.DigitalInOut(board.D13)

lcd_columns = 16
lcd_rows = 2

lcd = character_lcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows, lcd_backlight)
flower=Plant("flower",0,0)
# lcd.message=flower.getName()

import time
import adafruit_dht
import analogio
photocell = analogio.AnalogIn(board.A0)
dht_device = adafruit_dht.DHT22(board.D8)


while True:
    try:
        # Print the values to the serial port
        temperature_c = dht_device.temperature
        humidity = dht_device.humidity
        val= photocell.value
        volts = analogio.analog_voltage(photocell)
        # Convert to Fahrenheit
        temperature_f = temperature_c * (9/5) + 32

        print(f"Temp: {temperature_f:.1f}F / {temperature_c:.1f}C  Humidity: {humidity:.1f}%")
        print('Photocell value: {0} voltage: {1}V'.format(val, volts))

    except RuntimeError as e:
        # Reading a sensor is guaranteed to fail once in a while, retry
        print(f"Reading error: {e.args[0]}")
        time.sleep(2) # Wait at least 2 seconds between readings
        continue

    time.sleep(2) # Wait at least 2 seconds between readings

