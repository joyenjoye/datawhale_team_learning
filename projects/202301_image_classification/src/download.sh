zhihao_asset="https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification"
data_dir="data/image_net"

curl $zhihao_asset/dataset/meta_data/imagenet_class_index.csv -o $data_dir/meta_data/class_index.csv --create-dirs 

curl $zhihao_asset/test/watermelon1.jpg -o $data_dir/test/watermelon1.jpg --create-dirs 
curl $zhihao_asset/test/banana1.jpg -o $data_dir/test/banana1.jpg --create-dirs 
curl $zhihao_asset/test/cat1.jpg -o $data_dir/test/cat1.jpg --create-dirs 

# husky，source: https://www.pexels.com/zh-cn/photo/2853130/
curl $zhihao_asset/test/husky1.jpeg -o $data_dir/test/husky1.jpeg --create-dirs 

# cat dog，来源：https://unsplash.com/photos/ouo1hbizWwo
curl $zhihao_asset/test/cat_dog.jpg -o $data_dir/test/cat_dog.jpg --create-dirs 

# video
curl $zhihao_asset/test/video_2.mp4 -o $data_dir/test/video_2.mp4 --create-dirs 