import React from 'react';
import "./Contact.scss";
import { Form, Title} from './BasicComponents';

const contact_ofri = 'Ofri Shani - BSc in Computer Science'
const ofri_in = 'https://www.linkedin.com/in/ofri-shani-b63523123/'
const contact_maayan = 'Maayan Shoel - BSc in Computer Science and Linguistics'
const maayan_in= 'https://www.linkedin.com/in/maayan-shoel/'
const contact_rotem = 'Rotem Mitrany - BSc in Computer Science and Linguistics'
const rotem_in = 'https://www.linkedin.com/in/rotem-mitrany/'

export const Contact = () => {
  return (
    <div className='screen'>
      <Form>
        <Title title='Contact us :)' />
          <div className='Text'>
              <p>{contact_rotem}<br></br>
              <a href={rotem_in} target='_blank'>{rotem_in}</a></p>
              <p>{contact_ofri} <br></br>
              <a href={ofri_in} target='_blank'>{ofri_in}</a>
              </p>
              <p>{contact_maayan}<br></br>
              <a href={maayan_in} target='_blank'>{maayan_in}</a></p>
          </div>
      </Form>
    </div>
  );
};