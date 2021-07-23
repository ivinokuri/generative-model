import RuleBasedAnomaly as RBA
from Style import Configure as Conf

global options


def intersection_labeling(labeling, *labelings):
    new_labeling = []
    labelings = list(labelings)
    labelings.append(labeling)
    for i in range(len(labeling)):
        lbl = True
        for labeling in labelings:
            lbl = lbl and (labeling[i] == Conf.POSITIVE_LABEL)
        new_labeling.append(Conf.POSITIVE_LABEL if lbl else Conf.NEGATIVE_LABEL)
    return new_labeling

def apply_rba(dataset, rules):
    rba = RBA.RuleBasedAnomaly()
    rba.fit(rules)
    type = Conf.TRAINING
    #new_datasets = []

    rules_functions= [rba.transform_coverage_percentage_columns]
        # [rba.transform_same_digits_number
        #           rba.transform_negative_values, rba.transform_not_negative_values,
        #           rba.transform_coverage_percentage_columns, rba.transform_exactly_one_value,rba.transform_corresponding_columns]
    pred = 1
    for rule_function in rules_functions:
        pred = rule_function(type, dataset)
        if pred[0] == 0:
            print "Failed on {0}".format(rule_function.__name__)
            break

    # pred = rba.transform_same_digits_number(type, dataset, rules)
    # pred = intersection_labeling(pred, rba.transform_positive_values(type, dataset, rules))
    # pred = intersection_labeling(pred, rba.transform_negative_values(type, dataset, rules))
    # pred = intersection_labeling(pred, rba.transform_not_negative_values(type, dataset, rules))
    # pred = intersection_labeling(pred, rba.transform_coverage_percentage_columns(type, dataset, rules))
    # #pred = intersection_labeling(pred, rba.transform_corresponding_columns(type, dataset))
    # pred = intersection_labeling(pred, rba.transform_exactly_one_value(type, dataset, rules))
    # # removable_cols = rba.get_removable_columns()
    # #new_datasets.append(DS.Datasets.remove_columns(dataset, removable_cols))
    return pred
