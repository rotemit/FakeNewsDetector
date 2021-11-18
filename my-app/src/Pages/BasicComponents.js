import React from 'react';
import './BasicComponents.scss';
import info from '../info.svg'

export const Form = ({children}) => {
    return <div className='formWrapper'>{children}</div>
}

export const Loader = () => {
    return <div className='loader' />;
};

export const Title = ({title}) => {
    return <div className='title'>{title}</div>;
}

export const Input = ({label, type, value, name, placeholder, onChange, isToolip, title}) => {
    return (
        <div className='input'>
            <label className='label'>{label}</label>
            <input 
                className='inputText'
                type={type}
                value={value} 
                name={name} 
                placeholder={placeholder} 
                onChange={onChange} 
            />
            {isToolip &&
             <img title={title} src={info } className='info' /> }
        </div>
    )
}

export const SubmitButton = ({onSubmit}) => {
    return <button className='submitButton' onClick={onSubmit}>Submit</button>
}

export const Modal = ({ handleClose, show, text }) => {
    const showHideClassName = show ? "modal display-block" : "modal display-none";
  
    return (
      <div className={ showHideClassName}>
        <section className="modal-main">
          <div className='popupText'>{text}</div>
          <button className='popupButton' type="button" onClick={handleClose}>
            Close
          </button>
        </section>
      </div>
    );
  };