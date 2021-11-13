import React from 'react';
import "./About.scss";
import { Form, Title} from './BasicComponents';
// import React, { useState, useEffect } from 'react';

// import { Show_result } from "../Components/Show_result";

export const About = () => {
 
    return (
      <div className='screen'>
        <Form>
          <Title title='Welcome to our app!' />
          <p>Here a little explanation</p>
        </Form>
      </div>
    )
}