import datetime

remain = datetime.timedelta(minutes=2, seconds=25)
# 1 ~ 3.jpg -> 2:25
# 4 ~ 33.jpg -> 2:24
# 34 ~ 63.jpg -> 2:23
# 290 ~ 894.jpg -> finish_battle
first_time_change = 4
frame_rate = 30
time_to_finish = 290

for i in range(1, 894+1):
    if (i - first_time_change) % frame_rate == 0:
        # print(f"{i} remain time change!")
        remain -= datetime.timedelta(seconds=1)
    image_path = f"{str(i).zfill(6)}.jpg"
    # label = "finish" if i >= time_to_finish else remain #=> AttributeError: 'tuple' object has no attribute 'to'

    label = 0 if i >= time_to_finish else remain.seconds
    print(f"{image_path},{label}")

