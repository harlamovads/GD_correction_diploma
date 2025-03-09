cd data
tar -xvzf data.tar.gz
mkdir examtexts
mkdir rawfiles
cp -r /home/zlovoblachko/GD_correction_diploma/data/data/exam/Exam2014 /home/zlovoblachko/GD_correction_diploma/data/examtexts
cp -r /home/zlovoblachko/GD_correction_diploma/data/data/exam/Exam2015 /home/zlovoblachko/GD_correction_diploma/data/examtexts
cp -r /home/zlovoblachko/GD_correction_diploma/data/data/exam/Exam2016 /home/zlovoblachko/GD_correction_diploma/data/examtexts
cp -r /home/zlovoblachko/GD_correction_diploma/data/data/exam/Exam2017 /home/zlovoblachko/GD_correction_diploma/data/examtexts
cp -r /home/zlovoblachko/GD_correction_diploma/data/data/exam/Exam2019 /home/zlovoblachko/GD_correction_diploma/data/examtexts
cp -r /home/zlovoblachko/GD_correction_diploma/data/data/exam/Exam2020 /home/zlovoblachko/GD_correction_diploma/data/examtexts
find /home/zlovoblachko/GD_correction_diploma/data/examtexts -type f -print0 | xargs -0 mv -t /home/zlovoblachko/GD_correction_diploma/data/rawfiles
rm -rf data
rm -rf examtexts