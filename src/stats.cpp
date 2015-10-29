#include <numeric>
#include <cstdint>
#include <fstream>
#include <istream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "meta.h"

using namespace meta;

MAKE_NUMERIC_IDENTIFIER(student_id, uint64_t)

struct grade
{
    student_id id;
    class_label questions;
    class_label answers;
    class_label quality;
    class_label analysis;
    class_label clarity;
    class_label application;
    double overall;
};

double score(const class_label& cls)
{
    static std::unordered_map<class_label, double> scores = {{"N"_cl, 1.0},
                                                             {"B"_cl, 2.0},
                                                             {"C"_cl, 3.0},
                                                             {"P"_cl, 4.0},
                                                             {"E"_cl, 5.0}};

    return scores.at(cls);
}

double score(const grade& g)
{
    return (score(g.questions) + score(g.answers) + score(g.quality)
            + score(g.analysis) + score(g.clarity) + score(g.application))
           / 6.0;
}

std::istream& operator>>(std::istream& is, grade& g)
{
    std::string line;
    std::getline(is, line);
    if (!is)
        return is;
    std::stringstream ss{line};

    std::string val;
    std::getline(ss, val, ',');
    g.id = student_id{std::stoul(val)};

    std::getline(ss, val, ',');
    g.questions = class_label{val};

    std::getline(ss, val, ',');
    g.answers = class_label{val};

    std::getline(ss, val, ',');
    g.quality = class_label{val};

    std::getline(ss, val, ',');
    g.analysis = class_label{val};

    std::getline(ss, val, ',');
    g.clarity = class_label{val};

    std::getline(ss, val, ',');
    g.application = class_label{val};

    g.overall = score(g);

    return is;
}

void print_hist(const std::string& name,
                std::unordered_map<class_label, double>& hist, uint64_t count)
{
    std::cout << "Histogram for " << name << " (" << count << ")" << std::endl;

    std::cout << std::left << std::setw(12);
    std::cout << "Novice:" << hist[class_label{"N"}] / count << " ("
              << hist[class_label{"N"}] << ")" << std::endl;

    std::cout << std::left << std::setw(12);
    std::cout << "Beginner:" << hist[class_label{"B"}] / count << " ("
              << hist[class_label{"B"}] << ")" << std::endl;

    std::cout << std::left << std::setw(12);
    std::cout << "Competent:" << hist[class_label{"C"}] / count << " ("
              << hist[class_label{"C"}] << ")" << std::endl;

    std::cout << std::left << std::setw(12);
    std::cout << "Proficient:" << hist[class_label{"P"}] / count << " ("
              << hist[class_label{"P"}] << ")" << std::endl;

    std::cout << std::left << std::setw(12);
    std::cout << "Expert:" << hist[class_label{"E"}] / count << " ("
              << hist[class_label{"E"}] << ")" << std::endl;
    std::cout << std::endl;
}

int main()
{
    std::ifstream stream{"../data/Tuffy/tuffyrubric.csv"};
    std::string line;
    std::getline(stream, line);

    std::unordered_map<class_label, double> quest_hist;
    std::unordered_map<class_label, double> ans_hist;
    std::unordered_map<class_label, double> qual_hist;
    std::unordered_map<class_label, double> ana_hist;
    std::unordered_map<class_label, double> clar_hist;
    std::unordered_map<class_label, double> app_hist;

    uint64_t count{0};
    grade g;
    while (stream >> g)
    {
        ++count;
        ++quest_hist[g.questions];
        ++ans_hist[g.answers];
        ++qual_hist[g.quality];
        ++ana_hist[g.analysis];
        ++clar_hist[g.clarity];
        ++app_hist[g.application];

        std::cout << "overall: " << g.overall << std::endl;
    }

    print_hist("questions", quest_hist, count);
    print_hist("answers", ans_hist, count);
    print_hist("quality", qual_hist, count);
    print_hist("analysis", ana_hist, count);
    print_hist("clarity", clar_hist, count);
    print_hist("application", app_hist, count);
}
