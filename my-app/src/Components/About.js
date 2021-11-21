import React from 'react';
import "./About.scss";
import {AboutForm, Form, Title} from './BasicComponents';

const about_one = 'We are Maayan, Ofri, and Rotem, computer science students in Ben-Gurion University. We created this Hebrew Fake News Analyzer as our undergraduate final project.'
const about_two = 'Given a URL of either a post, an account, a group, or a page on Facebook, our Analyzer calculates three different parameters and according to them produces a grade between zero and one hundred, indicating the percentage of trustworthiness. A grade closer to zero indicates a high Fake News potential, and a grade closer to one hundred indicates a low Fake News potential, or a high level of trustworthiness. You can also enter text directly, to receive the SA and ML scores. If NumberOfPosts > 1, you will see the average of the scores from SA and ML for these posts.'
const about_three = 'The calculations consist of User Trust Value (UTV), Sentiment Analysis (SA), and Machine Learning (ML).'
const about_four = 'UTV takes into account parameters that have been shown to indicate a user\'s trustworthiness, in research conducted by our project advisor Nadav Voloch. These parameters consist of age of account (AUA), friendship duration (FD), total friends (TF), and mutual friends (MF). UTV is calculated only for accounts of friends from your first circle, and only if you are connected to Facebook. We have expanded UTV to be calculated on groups and pages as well, with the exclusion of the FD parameter. Also, if you scan a post in a group, UTV will be calculated for the writer of the post.'
const about_five = 'Sentiment analysis is performed similarly to this analysis in our predecessors\' project. Visit their GitHub repository for more detail: https://github.com/Simokod/Facebook-Profile-Analyzer'
const about_six = 'For ML we used a BERT-based pre-trained model: AlephBERT, published by ONLP Lab in Bar-Ilan University, and can be found here: https://github.com/OnlpLab/AlephBERT'
const about_seven = 'We added layers to this model and further trained it on our own dataset of Fake News in Hebrew. We constructed this dataset specifically for this project. You can find the dataset in our GitHub repo as well. This model is used to classify Fake News only on Covid-19 related posts. Thus, if a post is not covid-related, this test will be skipped.'
const about_eight = 'The average score of the calculated parameters is shown as well.'
const about_nine = 'Check out the Contact Us section to learn more about us.'
const about_ten = 'Happy Analyzing! :)'



// const aboutApp = `${about_one} \n ${about_two}`
const aboutApp = `${about_one} \r\n ${about_one}bla`
export const About = () => {
    return (
      <div className='screen'>
        <AboutForm>
          <Title title='About Fake News Analyzer'/>
          {/*<div className='text'>{aboutApp}</div>*/}
            <div className='text'>
            <p>{about_one}</p>
            <p>{about_two}</p>
            <p>{about_three}</p>
            <p>{about_four}</p>
            <p>{about_five}</p>
            <p>{about_six}</p>
            <p>{about_seven}</p>
            <p>{about_eight}</p>
            <p>{about_nine}</p>
            <p>{about_ten}</p>
            </div>
        </AboutForm>
      </div>
    );
};