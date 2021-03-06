import "./ScanPost.scss";
import React, { useState } from "react";
import {
  Loader,
  Form,
  SubmitButton,
  Input,
  Modal,
  Title,
} from "./BasicComponents";

export const ScanPost = () => {
  const [text, setText] = useState("");
  const [numOfPosts, setNumOfPosts] = useState(20);
  const [result, setResult] = useState({
    name: "Text",
    trust: null,
    machine: null,
    semantic: null,
  });
  const [isClicked, setIsClicked] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorText, setErrorText] = useState("");
  const [showModal, setShowModal] = useState(false);

  function Result() {
    return (
      <>
        <h3>{result.name} </h3>
        <table className="table">
          <tr>
            <th>Category</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Trust result</td>
            <td>
              {result.trust >= 0
                ? (Number(result.trust) * 100).toFixed(2) + "%"
                : "N/A"}
            </td>
          </tr>
          <tr>
            <td>Machine result</td>
            <td>
              {result.machine >= 0
                ? (Number(result.machine) * 100).toFixed(2) + "%"
                : "N/A"}
            </td>
          </tr>
          <tr>
            <td>Sentiment result</td>
            <td>
              {result.semantic >= 0
                ? (Number(result.semantic) * 100).toFixed(2) + "%"
                : "N/A"}
            </td>
          </tr>
        </table>
      </>
    );
  }

  async function handleSubmit() {
    setResult({ ...Result, semantic: -1 });
    setIsClicked(true);
    const response = await fetch("/scanPost", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: text, numOfPosts: numOfPosts }),
    });
    if (response.ok) {
      response.json().then((res) => {
        if (res.error !== undefined) {
          console.log(res.error);
          setErrorText(res.error);
          setHasError(true);
          setShowModal(true);
        } else {
          setText("");
          setNumOfPosts(20);
          setResult({
            name: res.name,
            trust: res.utv_result,
            machine: res.machineLearning_result,
            semantic: res.sentimentAnalyzer_result,
          });
        }
      });
    }
  }

  let total = 0;
  let devide = 0;
  if (result.trust !== -1) {
    total += Number(result.trust);
    devide++;
  }
  if (result.machine !== -1) {
    total += Number(result.machine);
    devide++;
  }
  if (result.semantic !== -1) {
    total += Number(result.semantic);
    devide++;
  }
  const finalAns = devide === 0 ? "N/A" : ((total / devide) * 100).toFixed(2);

  const clickCloseModal = () => {
    setIsClicked(false);
    setHasError(false);
    setShowModal(false);
  };
  const Toggle = () => {
    if (isClicked) {
      if (result.semantic >= 0) {
        return (
          <>
            <Result />
            <div className="total">Total: {finalAns}%</div>
          </>
        );
      } else {
        return <Loader />;
      }
    }
    return <div></div>;
  };

  const urlInfoText = `Two options:
  1. Enter text that you wish to examine. Maximum length: 511 words
  2. Enter the URL of either a Facebook post, account, group, or page that you wish to examine`;

  const numOfPostsInfoText = `Enter the number of posts you wish to scan and evaluate. 
  In single post scan, and in the simple text scan, this value will be ignored.`;

  return (
    <div className="screen">
      {hasError ? (
        <Modal
          handleClose={clickCloseModal}
          show={showModal}
          text={errorText}
        />
      ) : (
        <Form>
          <Title title="Check for realness" />
          <div className="fields">
            <Input
              label="Enter URL/text"
              type="textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              isToolip={true}
              title={urlInfoText}
            />
            <Input
              label="Number of posts"
              value={numOfPosts}
              onChange={(e) => setNumOfPosts(e.target.value)}
              isToolip={true}
              title={numOfPostsInfoText}
            />
            <SubmitButton onSubmit={handleSubmit} />
          </div>
          <Toggle />
        </Form>
      )}
    </div>
  );
};
