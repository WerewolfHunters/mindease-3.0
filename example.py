class Bmw:

    def __init__(self):
        self.num_doors =  4
        self.engine = "v6"
        self.is_sports = True
        self.speed = 0

    def move_forward(self):
        self.speed = self.speed + 10
        print("Increase speed by 10")
    
    def decrease(self):
        self.speed = self.speed - 5
        print("Decrease speed by 5")


if __name__=="__main__":
    awb_bmw = Bmw()
    kaif_bmw = Bmw()

    awb_bmw.engine = "v12"

    awb_bmw.move_forward()
    awb_bmw.move_forward()

    kaif_bmw.move_forward()

    print(f"Awwab speed: {awb_bmw.speed}, Kaif speed: {kaif_bmw.speed}")
