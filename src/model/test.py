output_size = 5
feature_num = 2

mask_ma1 = []
for sample_num in range(output_size):
    temp_list = [0.0]*output_size
    temp_list[sample_num] = 1.0
    mask_ma1.append(temp_list*feature_num)

print(mask_ma1)