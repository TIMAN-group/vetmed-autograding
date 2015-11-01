/**
 * @file active_l2r_assign.cpp
 * @author Chase Geigle
 *
 * This file runs the active learning to rank experiment. Instances are
 * assumed to be in libsvm format with regression style labels that
 * indicate their composite score (average across all rubrics).
 *
 * The application will then read all of these training instances in and
 * create a new binary dataset over *pairs* of instances, whose weights are
 * \f$x_i - x_j\f$ and whose label is \f$y_{ij} = sign(y_i - y_j)\f$. These
 * are then used as instances to learn a linear SVM model for pairwise
 * ranking.
 *
 * The supervision provided by the teacher, however, is now a real-valued
 * grade on an *assignment* basis, as opposed to a pairwise comparison
 * judgment.
 */

#include <cassert>
#include "cpptoml.h"
#include "classify/binary_dataset_view.h"
#include "classify/classifier/sgd.h"
#include "learn/loss/hinge.h"
#include "learn/sgd.h"
#include "index/eval/rank_correlation.h"
#include "index/make_index.h"
#include "regression/regression_dataset_view.h"
#include "util/progress.h"
#include "util/shim.h"

using namespace meta;

std::size_t pair_to_id(std::size_t i, std::size_t j, std::size_t n)
{
    return i * (n - 1) - i * (i - 1) / 2 + j - 1 - i;
}

std::pair<std::size_t, std::size_t> id_to_pair(std::size_t idx, std::size_t n)
{
    // do it a stupid way, doesn't matter
    auto rng = util::range<std::size_t>(1, n - 1);
    auto it = std::upper_bound(rng.begin(), rng.end(), idx,
                               [&](std::size_t val, std::size_t elem)
                               {
                                   return val < pair_to_id(elem, elem + 1, n);
                               });
    auto i = *it - 1;

    rng = util::range<std::size_t>(i + 1, n - 1);
    auto jit = std::upper_bound(rng.begin(), rng.end(), idx,
                                [&](std::size_t val, std::size_t elem)
                                {
                                    return val < pair_to_id(i, elem, n);
                                });
    auto j = *jit - 1;

    return {i, j};
}

int main(int argc, char** argv)
{
    logging::set_cerr_logging();
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " config.toml" << std::endl;
        return 1;
    }

    auto config = cpptoml::parse_file(argv[1]);
    auto f_idx = index::make_index<index::forward_index>(*config);

    auto al_config = config->get_table("active-learning-assign");
    auto num_seeds = al_config->get_as<int64_t>("num-seeds").value_or(5);
    auto max_train_size = static_cast<std::size_t>(
        al_config->get_as<int64_t>("max-train-size").value_or(50));

    auto doc_rng = util::range(0_did, doc_id{f_idx->num_docs() - 1});

    // load the dataset in as a regression dataset
    regression::regression_dataset reg_dset{
        f_idx, [&](doc_id did)
        {
            return *f_idx->metadata(did).get<double>("response");
        }};

    std::vector<double> reference_scores;
    reference_scores.reserve(reg_dset.size());
    std::transform(std::begin(reg_dset), std::end(reg_dset),
                   std::back_inserter(reference_scores),
                   [&](const learn::instance& inst)
                   {
                       return reg_dset.label(inst);
                   });

    // convert it to a binary ranking dataset by making a new instance for
    // every pair in the original
    struct binary_instance
    {
        learn::feature_vector weights;
        bool label;
    };
    std::vector<binary_instance> binary_instances;
    binary_instances.reserve(reg_dset.size() / 2 * (reg_dset.size() - 1));
    for (auto i = reg_dset.begin(); i != reg_dset.end(); ++i)
    {
        for (auto j = i + 1; j != reg_dset.end(); ++j)
        {
            auto label_diff = reg_dset.label(*i) - reg_dset.label(*j);
            binary_instances.push_back(
                {i->weights - j->weights, label_diff > 0});
        }
    }

    // construct the binary dataset from our transformation above
    // I *could* do this more intelligently, but it's not worth it here
    classify::binary_dataset bin_dset{std::begin(binary_instances),
                                      std::end(binary_instances),
                                      reg_dset.total_features(),
                                      [](binary_instance& inst)
                                      {
                                          return std::move(inst.weights);
                                      },
                                      [](binary_instance& inst)
                                      {
                                          return inst.label;
                                      }};

    // create a view over the original assignments and shuffle it to select
    // our seeds
    regression::regression_dataset_view rdv{reg_dset};
    rdv.shuffle();

    // create a view, DO NOT SHUFFLE
    classify::binary_dataset_view bdv{bin_dset};

    // create an empty view for the training set
    classify::binary_dataset_view train{bdv, bdv.end(), bdv.end()};
    regression::regression_dataset_view train_rdv{rdv, rdv.end(), rdv.end()};

    // insert all of the pairs from the seeds into the training set
    for (auto i = rdv.begin(); i != rdv.begin() + num_seeds; ++i)
    {
        for (auto j = i + 1; j != rdv.begin() + num_seeds; ++j)
        {
            if (i.index() < j.index())
            {
                train.add_by_index(
                    pair_to_id(i.index(), j.index(), rdv.size()));
            }
            else
            {
                train.add_by_index(
                    pair_to_id(j.index(), i.index(), rdv.size()));
            }
        }
        train_rdv.add_by_index(i.index());
    }
    assert(train_rdv.size() == num_seeds);
    assert(train.size() == num_seeds * (num_seeds - 1) / 2);

    printing::progress progress{" > Learning: ", bdv.size() - 1};
    std::ofstream results{"results-assign.csv"};
    results << "training-size,num-graded,NDPM\n";
    while (train_rdv.size() < rdv.size() && train_rdv.size() < max_train_size)
    {
        progress(train.size());
        // train a linear SVM on our learning-to-rank reduction
        classify::sgd svm{train, make_unique<learn::loss::hinge>(), {}};

        // get scores for all instances in the original data
        std::vector<double> system_scores;
        system_scores.reserve(reg_dset.size());
        std::transform(std::begin(reg_dset), std::end(reg_dset),
                       std::back_inserter(system_scores),
                       [&](const learn::instance& inst)
                       {
                           return svm.predict(inst.weights);
                       });

        // compute rank correlation measures
        index::rank_correlation corr{system_scores, reference_scores};
        results << train.size() << "," << train_rdv.size() << "," << corr.ndpm()
                << "\n";

        auto unlabled = rdv - train_rdv;
        assert(unlabled.size() + train_rdv.size() == rdv.size());

#if 0
        std::vector<double> scores;
        scores.reserve(unlabled.size());
#if 0
        // update the training set to include the assignment from the
        // unlabeled data that has the lowest confidence pair against any
        // assignment in the labeled data
        std::transform(
            std::begin(unlabled), std::end(unlabled),
            std::back_inserter(scores), [&](const learn::instance& inst)
            {
                auto it = std::min_element(
                    std::begin(train_rdv), std::end(train_rdv),
                    [&](const learn::instance& lhs, const learn::instance& rhs)
                    {
                        return std::abs(svm.predict(inst.weights - lhs.weights))
                               < std::abs(
                                     svm.predict(inst.weights - rhs.weights));
                    });
                return std::abs(svm.predict(inst.weights - it->weights));
            });
#else
        // update the training set to include the assignment from the
        // unlabeled data that has the lowest confidence total across all
        // pairs it would form with assignments in the labeled data
        std::transform(
            std::begin(unlabled), std::end(unlabled),
            std::back_inserter(scores), [&](const learn::instance& inst)
            {
                return std::accumulate(
                    std::begin(train_rdv), std::end(train_rdv), 0.0,
                    [&](double accum, const learn::instance& other)
                    {
                        return accum + std::abs(svm.predict(inst.weights
                                                            - other.weights));
                    });
            });
#endif

        auto it = std::min_element(std::begin(scores), std::end(scores));
        auto diff = it - std::begin(scores);
        auto inst_it = unlabled.begin() + diff;

        for (auto it = train_rdv.begin(); it != train_rdv.end(); ++it)
        {
            if (it.index() < inst_it.index())
            {
                train.add_by_index(
                    pair_to_id(it.index(), inst_it.index(), rdv.size()));
            }
            else
            {
                train.add_by_index(
                    pair_to_id(inst_it.index(), it.index(), rdv.size()));
            }
        }
        train_rdv.add_by_index(inst_it.index());
#elif 0
        // update the training set to include the pair of assignments that
        // is least confident under the current model
        //
        // this may add either one or two assignments to the training data

        auto test = bdv - train;
        auto it = std::min_element(
            std::begin(test), std::end(test),
            [&](const learn::instance& lhs, const learn::instance& rhs)
            {
                return std::abs(svm.predict(lhs.weights))
                       < std::abs(svm.predict(rhs.weights));
            });

        std::size_t x;
        std::size_t y;
        std::tie(x, y) = id_to_pair(it.index(), rdv.size());
        assert(pair_to_id(x, y, rdv.size()) == it.index());

        std::unordered_set<std::size_t> used;
        for (const auto& inst : train_rdv)
            used.insert(inst.id);

        if (used.find(x) == used.end())
        {
            for (auto it = train_rdv.begin(); it != train_rdv.end(); ++it)
            {
                if (it.index() < x)
                {
                    train.add_by_index(pair_to_id(it.index(), x, rdv.size()));
                }
                else
                {
                    train.add_by_index(pair_to_id(x, it.index(), rdv.size()));
                }
            }
            train_rdv.add_by_index(x);
        }

        if (used.find(y) == used.end())
        {
            for (auto it = train_rdv.begin(); it != train_rdv.end(); ++it)
            {
                if (it.index() < y)
                {
                    train.add_by_index(pair_to_id(it.index(), y, rdv.size()));
                }
                else
                {
                    train.add_by_index(pair_to_id(y, it.index(), rdv.size()));
                }
            }
            train_rdv.add_by_index(y);
        }
#else
        // randomly add a new question to the training set
        auto test = rdv - train_rdv;
        test.shuffle();

        for (auto it = train_rdv.begin(); it != train_rdv.end(); ++it)
        {
            if (it.index() < test.begin().index())
            {
                train.add_by_index(
                    pair_to_id(it.index(), test.begin().index(), rdv.size()));
            }
            else
            {
                train.add_by_index(
                    pair_to_id(test.begin().index(), it.index(), rdv.size()));
            }
        }
        train_rdv.add_by_index(std::begin(test).index());
#endif
    }

    return 0;
}
