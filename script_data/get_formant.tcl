#!/usr/local/ActiveTcl/bin/tclsh

### OSW: made this on the basis of HTS getf0.tcl script

#  ---------------------------------------------------------------  #
#           The HMM-Based Speech Synthesis System (HTS)             #
#                       HTS Working Group                           #
#                                                                   #
#                  Department of Computer Science                   #
#                  Nagoya Institute of Technology                   #
#                               and                                 #
#   Interdisciplinary Graduate School of Science and Engineering    #
#                  Tokyo Institute of Technology                    #
#                     Copyright (c) 2001-2007                       #
#                       All Rights Reserved.                        #
#                                                                   #
#  Permission is hereby granted, free of charge, to use and         #
#  distribute this software and its documentation without           #
#  restriction, including without limitation the rights to use,     #
#  copy, modify, merge, publish, distribute, sublicense, and/or     #
#  sell copies of this work, and to permit persons to whom this     #
#  work is furnished to do so, subject to the following conditions: #
#                                                                   #
#    1. The code must retain the above copyright notice, this list  #
#       of conditions and the following disclaimer.                 #
#                                                                   #
#    2. Any modifications must be clearly marked as such.           #
#                                                                   #
#  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,  #
#  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM    #
#  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL       #
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   #
#  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF        #
#  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE    #
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY        #
#  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  #
#  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTUOUS   #
#  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR          #
#  PERFORMANCE OF THIS SOFTWARE.                                    #
#                                                                   #
#  ---------------------------------------------------------------  #
#
# get_f0.tcl
#
# f0 extraction script using snack 
#
# Created  by Shinji SAKO  Mon Sep  1 18:54:35 JST 2003
# Modified by Heiga ZEN    Fri Nov  3 17:28:33 JST 2006

package require snack

set method ESPS
set maxpitch 400      
set minpitch 60       
set framelength 0.005 
set frameperiod 80    
set samplerate 16000  
set encoding Lin16    
set endian bigEndian 
set outputmode 0     
set targetfile ""
set outputfile ""

set arg_index $argc
set i 0
set j 0

set help [ format "pitch extract tool using snack library (= ESPS get_f0)\nUsage %s \[-H max_f0\] \[-L min_f0\] \[-s frame_length (in second)\] \[-p frame_length (in point)\] \[-r samplerate\] \[-l (little endian)\] \[-b (big endian)\] \[-o output_file\] \[-f0 (output in f0)] \[-lf0 (output in log f0)\] inputfile" $argv0 ]

while { $i < $arg_index } {
    switch -exact -- [ lindex $argv $i ] {
    -H {
        incr i
        set maxpitch [ lindex $argv $i ]
    }
    -L {
        incr i
        set minpitch [ lindex $argv $i ]
    }
    -s {
        incr i
        set framelength [ lindex $argv $i ]       
    }
    -p {
        incr i
        set frameperiod [ lindex $argv $i ]
        set j 1
    }
    -o {
        incr i
        set outputfile [ lindex $argv $i ]       
    }
    -r {
        incr i
        set samplerate [ lindex $argv $i ]       
    }
    -l {
        set endian littleEndian
    }
    -b {
        set endian bigEndian
    }
    -f0 {
        set outputmode 1
    }
    -lf0 {
        set outputmode 2
    }
    -h {
        puts stderr $help
        exit 1
    }
    default { set targetfile [ lindex $argv $i ] }
    }
    incr i
}

# framelength
if { $j == 1 } {
   set framelength [expr {double($frameperiod) / $samplerate}]
}

# if input file does not exist, exit program
if { $targetfile == "" } {
    puts stderr $help
    exit 0
}

snack::sound s 

# if input file is WAVE (RIFF) format, read it
if { [file isfile $targetfile ] && "[file extension $targetfile]" == ".wav"} {
    s read $targetfile
} else {
    s read $targetfile -fileformat RAW -rate $samplerate -encoding $encoding -byteorder $endian
}

# if output filename (-o option) is not specified, output result to stdout
set fd stdout

# extract f0 and output results
#set tmp [s formant ]
set tmp [s formant -numformants 4 -framelength $framelength ]
foreach line $tmp {
    #puts [lindex $line 0]
    puts $line
}

