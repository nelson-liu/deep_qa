# Datasets

At times, we have made changes to the default generated versions of various
datasets in order to make adding them into the pipeline. These changes are
documented here.

## NewsQA 

The default version of the NewsQA dataset (as of 1/19/17) is very difficult to
parse in our Scala data pipeline due to the irregular mixing of LF and CRLF line
endings. As a result, we have provided a file named
[`clean_newsqa.py`](../../../../../../../src/main/python/scripts/clean_newsqa.py)
in the Python
[`scripts/` directory](../../../../../../../src/main/python/scripts).
After generating the NewsQA dataset, please use the script to clean the data and
prepare it for use in the pipeline; run `python clean_newsqa.py -h` for more
information. You **must** use Python 2.x to run this script, as instructed by
[the NewsQA repo](https://github.com/Maluuba/newsqa#requirements).

Specifically, this script parses the NewsQA CSVs with the Pandas library and
replaces all the newlines (both LF and CRLF) with spaces (so each passage is one
contiguous line).

## Who Did What

The default version of the Who Did What dataset (as of 1/19/17) has non-breaking
spaces in the text (`&nbsp;`). The function we use to parse the XML,
`scala.xml.XML.loadString`, does not support these characters. Thus, run the
following command to replace the non-breaking spaces with normal spaces:

```
sed -i.bak 's/&nbsp;/ /g' <split>.xml
```

Where <split>.xml is one of the original generated Who Did What XML files. This
command uses the `sed` stream editor to replace all occurences of `&nbsp;` with
a space in place, and generates a backup of the original named
`<split>.xml.bak`.
