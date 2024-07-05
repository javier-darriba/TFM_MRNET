#ifndef CUFEAST_FSTOOLBOX_H
#define CUFEAST_FSTOOLBOX_H

#include <limits>

typedef std::vector <std::pair<int, double>> Results;

template<typename R>
class fs_criterion_params {
public:
    explicit fs_criterion_params(const R *class_mi) : class_mi(class_mi) {}

    fs_criterion_params(const R *class_mi, const R *feature_mi, unsigned int num_selected) : fs_criterion_params(
            class_mi) {
        this->feature_mi = feature_mi;
        this->num_selected = num_selected;
    }

    fs_criterion_params(const R *class_mi, const R *feature_mi, const R *feature_cmi, unsigned int num_selected)
            : fs_criterion_params(class_mi, feature_mi, num_selected) {
        this->feature_cmi = feature_cmi;
    }

    fs_criterion_params(const R *class_mi, const R *feature_mi, const R *feature_cmi, unsigned int num_selected,
                        double beta, double gamma) : fs_criterion_params(class_mi, feature_mi, feature_cmi,
                                                                         num_selected) {
        this->beta = beta;
        this->gamma = gamma;
    }

    const R *class_mi = nullptr;
    const R *feature_mi = nullptr;
    const R *feature_cmi = nullptr;

    unsigned int num_selected = 0;

    R beta = 0.0;
    R gamma = 0.0;
};

template<typename R>
using fs_criterion = R (*)(const fs_criterion_params<R>, unsigned int, unsigned int);

template<typename R>
void selectFeature(unsigned int num_features, bool *selected_features, Results &results, fs_criterion<R> score_f,
                   const fs_criterion_params<R> score_p) {
    unsigned int best_idx = 0;
    R best_score = -std::numeric_limits<R>::max();

    for (unsigned int i = 0; i < num_features; i++) {
        if (!selected_features[i]) {
            R current_score = score_f(score_p, num_features, i);

            if (current_score > best_score) {
                best_score = current_score;
                best_idx = i;
            }
        }
    }

    selected_features[best_idx] = true;
    results.emplace_back(best_idx, (double) best_score);
}

template<typename R>
R fsCriterionMIM(const fs_criterion_params<R> score_p, [[maybe_unused]] unsigned int num_features,
                 unsigned int current) {
    /*** MIM(f,c,S) = MI(f,c) ***/
    return score_p.class_mi[current];
}

template<typename R>
R fsCriterionMRMR(const fs_criterion_params<R> score_p, unsigned int num_features, unsigned int current) {
    /*** MRMR(f,c,S) = MI(f,c) - 1/|S| * SUM_{f_s in S} MI(f,f_s) ***/
    R selected_sc = score_p.class_mi[current];
    for (unsigned int i = 0; i < score_p.num_selected; i++) {
        selected_sc += -score_p.feature_mi[i * num_features + current] / (double) score_p.num_selected;
    }

    return selected_sc;
}

template<typename R>
R fsCriterionJMI(const fs_criterion_params<R> score_p, unsigned int num_features, unsigned int current) {
    /*** JMI(f,c,S) = SUM_{f_s in S} MI(f,f_s) ***/
    R selected_sc = 0.0;
    for (unsigned int i = 0; i < score_p.num_selected; i++) {
        selected_sc += score_p.feature_mi[i * num_features + current];
    }

    return selected_sc;
}

template<typename R>
R fsCriterionDISR(const fs_criterion_params<R> score_p, unsigned int num_features, unsigned int current) {
    /*** DISR(f,c,S) = SUM_{f_s in S} MI(f,f_s)/JE(f,f_s) ***/
    // MI/JE comes already computed in *feature_mi*
    R selected_sc = 0.0;
    for (unsigned int i = 0; i < score_p.num_selected; i++) {
        selected_sc += score_p.feature_mi[i * num_features + current];
    }

    return selected_sc;
}

#endif //CUFEAST_FSTOOLBOX_H