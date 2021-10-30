import { Test, TestingModule } from '@nestjs/testing';
import { HttpStatus, INestApplication } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../src/app/app.module';

const API_HOST = process.env.API_HOST || 'https://kata.getmansa.com';

describe('Kanedama', () => {
  let app: INestApplication;

  beforeEach(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  describe('GET /answer', () => {
    it('should respond with the correct answer', () =>
      request(app.getHttpServer())
        .get('/answer')
        .expect(HttpStatus.OK)
        .expect('Content-Type', /json/)
        .expect(({ body: applicantAnswer }) =>
          expect(applicantAnswer).toMatchObject({
            '6_month_average_income': expect.any(Number),
            min_balance: expect.any(Number),
            max_balance: expect.any(Number),
            '3_years_activity': expect.any(Boolean),
          }),
        )
        .then(({ body: applicantAnswer }) =>
          request(API_HOST)
            .post('/answer')
            .send(applicantAnswer)
            .expect(HttpStatus.CREATED)
            .then(({ body: { answer: validationStatus } }) =>
              expect(validationStatus).toMatch(/^Congratz!/),
            ),
        ));
  });
});
