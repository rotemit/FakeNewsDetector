import React from 'react';
import "./About.scss";
import { Form, Title} from './BasicComponents';
// import React, { useState, useEffect } from 'react';

// import { Show_result } from "../Components/Show_result";
const aboutApp = `We are Ofri, Maayan, and Rotem, computer science students in Ben-Gurion University. We created this Hebrew Fake News Analyzer as our undergraduate final project.
      Given a URL of either a post, an account, a group, or a page on Facebook, our Analyzer calculates three different parameters and according to them produces a grade between zero and one hundred, indicating the percentage of potential for spreading real news. A grade closer to zero indicates a high potential of spreading Fake News, and a grade closer to one hundred indicates potential for spreading real news.
      The calculations consist of User Trust Value (UTV), Sentiment Analysis, and Machine Learning (ML). For ML we used a BERT-based pre-trained model: AlephBERT, published by ONLP Lab in Bar-Ilan University, and can be found here:
      https://github.com/OnlpLab/AlephBERT
      We invite you to check out our final report for more information and detail.
      Check out the Contact Us section to learn more about us.`
export const About = () => {
 
    return (
      <div className='screen'>
        <Form>
          <Title title='About Fake News Analayzer!' />
          <div className='text'>{aboutApp}</div>
        </Form>
      </div>
    )
}