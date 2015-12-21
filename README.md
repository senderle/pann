pann
====

I am sad to say that this project has been more-or-less obsoleted by 
[Keras](keras.io). If, that is, it was ever non-obsolete to begin
with. But I hope that people will still find it useful for study or
play.
_______________________________________________________________

A simple feed forward neural network trainer. 

My goal is to create an application for training feed forward networks on 
large quantitites of data. Right now it can handle several million training
instances in a reasonable way. I want it to handle several hundred million, 
and to train across multiple computers in a cluster. Those goals are a bit 
more challenging! But this already deals with interesting, mid-size problems
pretty well. It's able to train a decent character-level language model in a 
week. It's not as good as this: 

http://www.icml-2011.org/papers/524_icmlpaper.pdf

But it produces some pretty reasonable word sequences, and since I don't
know how to write a Hessian free optimizer (yet!) I think it's good enough
for now. 

If you use this, get in touch! I'd love to work on making this as feature-
complete as possible, but I don't know what would be useful to others. 

scott.enderle@gmail.com
