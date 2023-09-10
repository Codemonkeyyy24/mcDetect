#!/usr/bin/perl-w
#use strict;
die  "Usage: $0 <input> <out_put>\n" unless (@ARGV==2);
open IN,"$ARGV[0]" or die $!;
open (data1, ">./data1.txt") or die $!;
open (OUT, ">$ARGV[1]") or die $!;
my %hash;
my @a;
while(<IN>){
	chomp; 
	@a=split; 
	$hash{$a[1]."\t".$a[3]."\t".$a[7]}++;
} 
foreach my $key(keys %hash){
	print data1 $key."\t".$hash{$key}."\n";
}

open (IN2, "data1.txt") or die $!;
my %hash1;
my %hash2;
my %hash0;
my @b;
while(<IN2>){
	chomp; 
	@b=split;
	$hash0{$b[0]."\t".$b[1]}=0; 
	if($b[2]=~/C/){
		$hash1{$b[0]."\t".$b[1]}=$b[2]."\t".$b[3];
	} 
	if($b[2]=~/M/){
		$hash2{$b[0]."\t".$b[1]}=$b[2]."\t".$b[3];
	}
} 
foreach my $key(keys %hash0){
	if(exists $hash1{$key} && exists $hash2{$key}){
		print OUT $key."\t".$hash1{$key}."\t".$hash2{$key}."\n";
	} 
	if(exists $hash1{$key} && !exists $hash2{$key}){
		print OUT $key."\t".$hash1{$key}."\t"."M"."\t"."0"."\n";
	}
	if(!exists $hash1{$key} && exists $hash2{$key}){
		print OUT $key."\t"."C"."\t"."0"."\t".$hash2{$key}."\n";
	}
}
system("rm -f data1.txt");
close IN1;
close IN2;
close OUT;
