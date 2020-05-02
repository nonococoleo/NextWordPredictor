for n in {1..4}
do
    for train in corpus/train_all.pkl corpus/train_business.pkl corpus/train_enter.pkl corpus/train_pol.pkl corpus/train_sport.pkl corpus/train_tech.pkl
    do
        for test in corpus/test_all.pkl corpus/test_business.pkl corpus/test_enter.pkl corpus/test_pol.pkl corpus/test_sport.pkl corpus/test_tech.pkl
        do
        python3 NGram.py --N $n --training_file $train --test_file $test --output_file output.txt
        done
    done
done
