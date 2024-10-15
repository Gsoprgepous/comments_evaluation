# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PLkq8WG-llHYDcG55l2dIorXmJ0awNVi
"""

from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(widget=forms.Textarea(attrs={
        'placeholder': 'Enter your review here',
        'rows': 4,
        'cols': 50
    }))