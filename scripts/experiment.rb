require 'descriptive_statistics'

if ARGV.length < 2
    puts "Usage: #{$0} file percent-train"
    exit 1
end

baseline = []
svor = []
training = []

10.times do
    lines = File.readlines(ARGV[0]).shuffle

    train_len = [1, (lines.length * ARGV[1].to_f).round].max
    puts "Training set size: #{train_len}"

    lbl_hist = {}
    File.open("#{ARGV[0]}.train", "w") do |f|
        for i in (0..train_len-1)
            lbl = lines[i].split[0]
            if lbl_hist[lbl]
                lbl_hist[lbl] += 1
            else
                lbl_hist[lbl] = 1
            end
            f.puts lines[i]
        end
    end

    puts lbl_hist.to_a.sort{|a, b| a[0] <=> b[0]}.inspect
    majclass = lbl_hist.keys.sort{|a, b| lbl_hist[b] <=> lbl_hist[a]}[0]

    correct = 0.0
    total = 0.0
    error = 0.0
    puts "Testing set size: #{lines.length - train_len}"

    lbl_hist = {}
    File.open("#{ARGV[0]}.test", "w") do |f|
        for i in ((train_len)..lines.length-1)
            total += 1
            lbl = lines[i].split[0]
            correct += 1 if lbl == majclass
            error += (majclass.to_i - lbl.to_i).abs

            if lbl_hist[lbl]
                lbl_hist[lbl] += 1
            else
                lbl_hist[lbl] = 1
            end

            f.puts lines[i]
        end
    end
    puts lbl_hist.to_a.sort{|a, b| a[0] <=> b[0]}.inspect
    puts "\n"

    baseline_acc = correct / total
    baseline_mae = error / total

    baseline.push(baseline_mae)

    puts "Majority class (in training = #{majclass}) baseline performance:"
    puts "Accuracy (0-1): #{correct / total}"
    puts "Mean absolute error: #{error / total}"
    puts "\n"

    puts "SVOR results:"
    puts `./svm-train -s 5 -t 0 #{ARGV[0]}.train #{ARGV[0]}.model`
    #puts `./svm-train -s 5 #{ARGV[0]}.train #{ARGV[0]}.model`
    results = `./svm-predict #{ARGV[0]}.test #{ARGV[0]}.model /dev/null`
    puts results
    puts "\n"

    svor_acc = results.split("Accuracy = ")[1].split("% )")[0].to_f / 100
    svor_mae = results.split("error = ")[1].split(" (regression)")[0].to_f

    svor.push(svor_mae)

    train_res = `./svm-predict #{ARGV[0]}.train #{ARGV[0]}.model /dev/null`
    train_mae = train_res.split("error = ")[1].split(" (regression)")[0].to_f

    training.push(train_mae)

    if svor_acc > baseline_acc
        puts "SVOR is better than baseline by #{svor_acc - baseline_acc} ACC"
    else
        puts "Baseline is better than SVOR by #{baseline_acc - svor_acc} ACC"
    end

    if svor_mae < baseline_mae
        puts "SVOR is better than baseline by #{baseline_mae - svor_mae} MAE"
    else
        puts "Baseline is better than SVOR by #{svor_mae - baseline_mae} MAE"
    end
end

puts "\n"
puts "Baseline average MAE: #{baseline.mean}, stddev: #{baseline.standard_deviation}"
puts "SVOR average MAE: #{svor.mean}, stddev: #{svor.standard_deviation}"
puts "SVOR average MAE on training set: #{training.mean}, stddev = #{training.standard_deviation}"
puts "\n"

puts "#{baseline.mean.round(4)} $\\pm$ #{baseline.standard_deviation.round(4)}\\\\"
puts "#{svor.mean.round(4)} $\\pm$ #{svor.standard_deviation.round(4)}"
