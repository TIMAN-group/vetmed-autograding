/**
 * @file active_l2r.cpp
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
 * Instances are chosen using uncertainty sampling where the measure of
 * uncertainty is the distance from the decision boundary. One instance at
 * a time is chosen, and the model is re-fit using the new training
 * instances.
 */

#include "cpptoml.h"
#include "classify/binary_dataset_view.h"
#include "classify/classifier/sgd.h"
#include "learn/loss/hinge.h"
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

    auto al_config = config->get_table("active-learning");
    auto num_seeds = al_config->get_as<int64_t>("num-seeds").value_or(1);
    auto max_train_size = static_cast<std::size_t>(
        al_config->get_as<int64_t>("max-train-size").value_or(1000));

    std::cout << "num instances: " << f_idx->num_docs() << std::endl;
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

    // create a view and shuffle it
    classify::binary_dataset_view bdv{bin_dset};
    bdv.shuffle();

    // select our seeds into the training set
    classify::binary_dataset_view train{bdv, bdv.begin(),
                                        bdv.begin() + num_seeds};

    printing::progress progress{" > Learning: ", bdv.size() - 1};
    std::ofstream results{"results.csv"};
    results << "training-size,num-distinct,NDPM\n";
    while (train.size() < bdv.size() && train.size() < max_train_size)
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

        std::unordered_set<std::size_t> used;
        for (const auto& inst : train)
        {
            std::size_t x;
            std::size_t y;
            std::tie(x, y) = id_to_pair(inst.id, reg_dset.size());

            used.insert(x);
            used.insert(y);
        }

        // compute rank correlation measures
        index::rank_correlation corr{system_scores, reference_scores};
        results << train.size() << "," << used.size()
                << "," << corr.ndpm() << "\n";

        auto test = bdv - train;
#if 1
        // update training set to include least confident pairwise example
        // in the "unlabeled" data
        auto it = std::min_element(
            std::begin(test), std::end(test),
            [&](const learn::instance& lhs, const learn::instance& rhs)
            {
                return std::abs(svm.predict(lhs.weights))
                       < std::abs(svm.predict(rhs.weights));
            });
#else
        // add a random point to the training set
        test.shuffle();
        auto it = test.begin();
#endif

        train.add_by_index(it.index());
    }

    return 0;
}
