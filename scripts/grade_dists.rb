require 'descriptive_statistics'

labels = []
File.foreach(ARGV[0]) do |l|
    labels.push(l.split[0].to_i)
end

puts "#{labels.mean.round(4)} $\\pm$ #{labels.standard_deviation.round(4)}"
