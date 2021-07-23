def output_to_csv(lines = None):
    if lines == None:
        with open('manipulation_gripping_v1_time_0.125.txt') as f:
            lines = f.readlines()
    else:
        lines = lines.split('\n')

    faulty_runs_negative = []
    non_faulty_runs_negative = []

    faulty_runs_positive = []
    non_faulty_runs_positive = []

    bigger_than_max_most_negative_list = []
    bigger_than_max_most_negative_in_negative_zone_list = []
    min_accuracy_training = []
    less_than_min_accuracy_list = []

    longest_average_training_list = []
    longest_average_positive_list = []
    longest_average_negative_list = []
    below_min_accuracy_negative_zone = 0
    min_accuracy_in_negative_zone = 1.0

    zone = None
    bigger_than_max_most_negative = None
    bigger_than_max_most_negative_in_negative_zone = None
    less_than_min_accuracy = None

    average = None
    minimum_accuracy = None
    max_threshold = None
    accuracy_min_positives = []
    title = None
    name = None
    time = None

    result = ''


    for line in lines:
        if line.startswith('available runs/time '):
            line_split = line.split('available runs/time ')[1]
            line_split = line_split.split(',')[0]
            title = line_split.split(' ')[0]
            name = line_split.split('/')[1]
            time = line_split.split('/')[2] + " " + line_split.split('/')[3]

        if line.startswith('-----training sets-----'):
            zone = 'training_sets'
            max_most_negative = 0
            if bigger_than_max_most_negative is not None:
                bigger_than_max_most_negative_list.append(bigger_than_max_most_negative)
            if less_than_min_accuracy is not None:
                less_than_min_accuracy_list.append(less_than_min_accuracy)
            line_split = line.split('longest average = ')[1]
            longest_average_training_list.append(float(line_split.split(',')[0]))

        elif line.startswith('-----positive sets-----'):
            zone = 'positive_sets'
            bigger_than_max_most_negative = 0
            less_than_min_accuracy = 0
            line_split = line.split('faulty runs = ')[1]
            faulty_runs_positive.append(float(line_split.split(',')[0]))
            line_split = line.split('non-faulty runs = ')[1]
            non_faulty_runs_positive.append(float(line_split.split(',')[0]))
            line_split = line.split('longest average = ')[1]
            longest_average_positive_list.append(float(line_split.split(',')[0]))

        elif line.startswith('-----negative sets-----'):
            zone = 'negative_sets'
            bigger_than_max_most_negative_in_negative_zone = 0
            line_split = line.split('longest average = ')[1]
            longest_average_negative_list.append(float(line_split.split(',')[0]))
            line_split = line.split('faulty runs = ')[1]
            faulty_runs_negative.append(float(line_split.split(',')[0]))
            line_split = line.split('non-faulty runs = ')[1]
            non_faulty_runs_negative.append(float(line_split.split(',')[0]))
            line_split = line.split('max threshold = ')[1]
            max_threshold = int(line_split.split(',')[0])

            if bigger_than_max_most_negative is not None:
                bigger_than_max_most_negative_list.append(bigger_than_max_most_negative)
            if less_than_min_accuracy is not None:
                less_than_min_accuracy_list.append(less_than_min_accuracy)

        elif '.csv' in line:
            line_split = line.split('most negative: ')[1]
            most_negative = int(line_split.split(' ')[0])

            line_split = line.split('accuracy: ')[1]
            accuracy = float(line_split.split(' ')[0])

            if zone == 'training_sets':
                if most_negative > max_most_negative:
                    max_most_negative = most_negative
            elif zone =='positive_sets':
                if most_negative > max_most_negative:
                    bigger_than_max_most_negative += 1
                if accuracy < min_accuracy_training[-1]:
                    less_than_min_accuracy += 1
            elif zone == 'negative_sets':
                if most_negative > max_most_negative:
                    bigger_than_max_most_negative_in_negative_zone += 1
                if '(below min accuracy)' in line:
                    below_min_accuracy_negative_zone += 1
                if accuracy < min_accuracy_in_negative_zone:
                    min_accuracy_in_negative_zone = accuracy

        elif line.startswith('accuarcy min trainings: '):
            line_split = line.split('accuarcy min trainings: ')[1]
            min_accuracy_training.append(float(line_split.split(',')[0]))
            line_split = line_split.split('accuarcy min positives: ')[1]
            accuracy_min_positives.append(float(line_split.split(',')[0]))

        if line.startswith('average: '):
            line_split = line.split('average: ')[1]
            average = float(line_split.split(',')[0])

        if line.startswith('minimum accuracy: '):
            line_split = line.split('minimum accuracy: ')[1]
            minimum_accuracy = float(line_split.split(',')[0])

    if bigger_than_max_most_negative is not None:
        bigger_than_max_most_negative_list.append(bigger_than_max_most_negative)
    if less_than_min_accuracy is not None:
        less_than_min_accuracy_list.append(less_than_min_accuracy)
    if bigger_than_max_most_negative_in_negative_zone is not None:
        bigger_than_max_most_negative_in_negative_zone_list.append(bigger_than_max_most_negative_in_negative_zone)

    print "title: " + title + " - " + name + " - " + time
    print "faulty runs     #1: " + str(faulty_runs_positive[0])
    print "most negative   #1: " + str(bigger_than_max_most_negative_list[0])
    print "min accuracy    #1: " + str(min_accuracy_training[0])
    print "accuracy faults #1: " + str(less_than_min_accuracy_list[0])
    print "faulty runs     #2: " + str(faulty_runs_positive[1])
    print "most negative   #2: " + str(bigger_than_max_most_negative_list[1])
    print "min accuracy    #2: " + str(min_accuracy_training[1])
    print "accuracy faults #2: " + str(less_than_min_accuracy_list[1])
    print "faulty runs     #3: " + str(faulty_runs_positive[2])
    print "most negative   #3: " + str(bigger_than_max_most_negative_list[2])
    print "min accuracy    #3: " + str(min_accuracy_training[2])
    print "accuracy faults #3: " + str(less_than_min_accuracy_list[2])
    print "faulty runs     #4: " + str(faulty_runs_positive[3])
    print "most negative   #4: " + str(bigger_than_max_most_negative_list[3])
    print "min accuracy    #4: " + str(min_accuracy_training[3])
    print "accuracy faults #4: " + str(less_than_min_accuracy_list[3])
    print "average success:    " + str(average)
    if minimum_accuracy is not None:
        print "minimum accuarcy:   " + str(minimum_accuracy)
        print "average longest training: " + str(longest_average_training_list[4])
        print "average longest positive: " + str(longest_average_positive_list[4])
        print "average longest negative: " + str(longest_average_negative_list[0])
        print "max threshold training: " + str(max_threshold)
        print "most negative training: " + str(max_most_negative)
        print "min accuarcy training: " + str(min_accuracy_training[4])
        print "faulty runs positive: " + str(faulty_runs_positive[4]) + " (" + str(faulty_runs_positive[4] + non_faulty_runs_positive[4]) + ")"
        print "most negative positive: " + str(bigger_than_max_most_negative_list[4]) + " (" + str(faulty_runs_positive[4] + non_faulty_runs_positive[4]) + ")"
        print "min accuarcy positive: " + str(accuracy_min_positives[4])
        print "min accuracy positive faults: " + str(less_than_min_accuracy_list[4]) + " (" + str(faulty_runs_positive[4] + non_faulty_runs_positive[4]) + ")"
        print "faulty runs negative: " + str(faulty_runs_negative[0]) + " (" + str(faulty_runs_negative[0] + non_faulty_runs_negative[0]) + ")"
        print "faulty most negative - negative: " + str(bigger_than_max_most_negative_in_negative_zone_list[0]) +  " (" + str(faulty_runs_negative[0] + non_faulty_runs_negative[0]) + ")"
        print "faulty min accuarcy negative: " + str(below_min_accuracy_negative_zone) + " (" + str(faulty_runs_negative[0] + non_faulty_runs_negative[0]) + ")"
        print "min accuarcy negative: " + str(min_accuracy_in_negative_zone)

    if minimum_accuracy is None:
        result = title + " - " + name + " - " + time + "\t" + str(faulty_runs_positive[0])\
              + "\t" + str(bigger_than_max_most_negative_list[0]) \
              + "\t" + str(min_accuracy_training[0]) + "\t" + str(less_than_min_accuracy_list[0]) \
              + "\t" + str(faulty_runs_positive[1]) + "\t" + str(bigger_than_max_most_negative_list[1]) \
              + "\t" + str(min_accuracy_training[1]) + "\t" + str(less_than_min_accuracy_list[1]) \
              + "\t" + str(faulty_runs_positive[2]) + "\t" + str(bigger_than_max_most_negative_list[2]) \
              + "\t" + str(min_accuracy_training[2]) + "\t" + str(less_than_min_accuracy_list[2]) \
              + "\t" + str(faulty_runs_positive[3]) + "\t" + str(bigger_than_max_most_negative_list[3]) \
              + "\t" + str(min_accuracy_training[3]) + "\t" + str(less_than_min_accuracy_list[3]) \
              + "\t" + str(average)
    else:
        result =  title + " - " + name + " - " + time + "\t" + str(faulty_runs_positive[0])\
              + "\t" + str(bigger_than_max_most_negative_list[0]) \
              + "\t" + str(min_accuracy_training[0]) + "\t" + str(less_than_min_accuracy_list[0]) \
              + "\t" + str(faulty_runs_positive[1]) + "\t" + str(bigger_than_max_most_negative_list[1]) \
              + "\t" + str(min_accuracy_training[1]) + "\t" + str(less_than_min_accuracy_list[1]) \
              + "\t" + str(faulty_runs_positive[2]) + "\t" + str(bigger_than_max_most_negative_list[2]) \
              + "\t" + str(min_accuracy_training[2]) + "\t" + str(less_than_min_accuracy_list[2]) \
              + "\t" + str(faulty_runs_positive[3]) + "\t" + str(bigger_than_max_most_negative_list[3]) \
              + "\t" + str(min_accuracy_training[3]) + "\t" + str(less_than_min_accuracy_list[3]) \
              + "\t" + str(average) + "\t" + str(minimum_accuracy)+ "\t" + str(longest_average_training_list[4])\
              + "\t" + str(longest_average_positive_list[4])+ "\t" + str(longest_average_negative_list[0])\
              + "\t" + str(max_threshold) + "\t" + str(max_most_negative) + "\t" + str(min_accuracy_training[4])\
              + "\t" + str(faulty_runs_positive[4]) + " (" + str(faulty_runs_positive[4] + non_faulty_runs_positive[4])\
              + ")" + "\t" + str(bigger_than_max_most_negative_list[4])\
              + " (" + str(faulty_runs_positive[4] + non_faulty_runs_positive[4]) + ")" \
              + "\t" + str(accuracy_min_positives[4]) + "\t" + str(less_than_min_accuracy_list[4]) + " (" + str(faulty_runs_positive[4] + non_faulty_runs_positive[4])\
              + ")"  + "\t" + str(faulty_runs_negative[0]) + " (" + str(faulty_runs_negative[0] + non_faulty_runs_negative[0])\
              + ")" + "\t" + str(bigger_than_max_most_negative_in_negative_zone_list[0]) +  " (" + str(faulty_runs_negative[0] + non_faulty_runs_negative[0])\
              + ")" + "\t" + str(below_min_accuracy_negative_zone) + " (" + str(faulty_runs_negative[0] + non_faulty_runs_negative[0])\
              + ")" "\t" + str(min_accuracy_in_negative_zone)
    print result
    return result

if __name__ == '__main__':
    output_to_csv()